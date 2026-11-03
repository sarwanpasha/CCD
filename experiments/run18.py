import sys,os,json,time,itertools
sys.path.insert(0,'../src')
import numpy as np
from scipy.optimize import minimize
exec(open('run10.py').read().split('def fsbound')[0])
T0=time.time()
def lg(*a): print(*a,flush=True)
z=np.load('cache_lc.npz'); DAT={k:z[k] for k in z.files}
TASKS=['arc','winogrande','hellaswag','mmlu']
def rhoS(G,x0=None):
  n,k=G.shape; wn=1.0/n
  def F(lam):
    zz=G@lam; mz=zz.max(); e=np.exp(zz-mz); s=e.sum()
    return mz+np.log(s*wn), (G*e[:,None]).sum(0)/s
  r=minimize(F,np.zeros(k) if x0 is None else np.asarray(x0,float),jac=True,method='L-BFGS-B',
    bounds=[(0.0,None)]*k,options={'ftol':1e-16,'gtol':1e-13,'maxiter':400})
  return -float(r.fun),r.x
BB={'arc':400,'winogrande':400,'hellaswag':150,'mmlu':150}
RK=[]
for reg in ['raw','acc']:
  for tk in TASKS:
    A=DAT[('C_' if reg=='acc' else 'L_')+tk]; cl=DAT['cl_'+tk]; n=A.shape[1]
    LS=(1.0-A) if reg=='acc' else A
    mn=LS.mean(axis=1); rank={int(j):int(i) for i,j in enumerate(np.argsort(mn))}
    for ti in range(6):
      riv=[j for j in range(6) if j!=ti]
      G=np.stack([LS[ti]-LS[j] for j in riv],axis=1)
      ahead=int((G.mean(axis=0)>0).sum())
      pw=[]
      for j in range(5):
        v,_=rhoS(G[:,[j]]); pw.append(float(v))
      for r in [1,2,3]:
        base={'reg':reg,'task':tk,'target':SHORT[MODELS[ti]],'r':r,'n':n,'rank':rank[ti],
          'ahead':ahead,'rivals':[SHORT[MODELS[j]] for j in riv],'pairwise':pw}
        if ahead>=r:
          base.update({'rho':0.0,'note':'already_out','ci':None}); RK.append(base)
          lg('%-3s %-11s tgt=%-4s r=%d ALREADY_OUT ahead=%d'%(reg,tk,SHORT[MODELS[ti]],r,ahead)); continue
        subs=list(itertools.combinations(range(5),r)); vals=[]
        for S in subs:
          v,lm=rhoS(G[:,list(S)]); vals.append((float(v),S,lm))
        vals.sort(key=lambda x:x[0]); best=vals[0]; cands=vals[:2]
        act=int((best[2]>1e-9).sum())
        B=BB[tk]; rng=np.random.default_rng(101+r); out=np.empty(B)
        uc=np.unique(cl); idxs=[np.where(cl==u)[0] for u in uc] if len(uc)>1 else None
        for b in range(B):
          ii=np.concatenate([idxs[j] for j in rng.integers(0,len(idxs),len(idxs))]) if idxs is not None else rng.integers(0,n,n)
          Gb=G[ii]; mv=np.inf
          for v0,S,lm in cands:
            vv,_=rhoS(Gb[:,list(S)],x0=lm); mv=min(mv,vv)
          out[b]=mv
        lo=float(np.percentile(out,2.5)); hi=float(np.percentile(out,97.5))
        base.update({'rho':best[0],'subset':[base['rivals'][j] for j in best[1]],
          'lam':[float(x) for x in best[2]],'active':act,'ci':[lo,hi],'boot_sd':float(out.std(ddof=1)),
          'hw_over_rho':float((hi-lo)/2.0/best[0]) if best[0]>0 else None,
          'second':vals[1][0] if len(vals)>1 else None,
          'min_pw':float(min(pw)),'gap_to_minpw':float(best[0]/min(pw)) if min(pw)>0 else None,
          'minpw_rival':base['rivals'][int(np.argmin(pw))],
          'minpw_in_subset':bool(int(np.argmin(pw)) in best[1])})
        RK.append(base)
        lg('%-3s %-11s tgt=%-4s r=%d rho=%.8f CI=[%.8f,%.8f] hw/rho=%.4f act=%d S=%s minpw=%.8f(%s in=%d) ratio=%.4f %.0fs'%(
          reg,tk,SHORT[MODELS[ti]],r,best[0],lo,hi,(hi-lo)/2.0/best[0] if best[0]>0 else -1,act,
          '+'.join(base['subset']),min(pw),base['minpw_rival'],int(base['minpw_in_subset']),
          base['gap_to_minpw'] if base['gap_to_minpw'] else -1,time.time()-T0))
json.dump({'rows':RK},open('out/run18a.json','w'))
lg('RANKR done rows=%d %.0fs'%(len(RK),time.time()-T0))
LGD=float(np.log(20.0)); NS=300
def seif(h,b):
  zz=b*h; mz=zz.max(); e=np.exp(zz-mz); Z=e.mean(); phi=1.0-e/Z
  return float(phi.std(ddof=1)/np.sqrt(len(h)))
SP=[]
for reg in ['raw','acc']:
  A=DAT[('C_' if reg=='acc' else 'L_')+'mmlu']; cl=DAT['cl_mmlu']; n=A.shape[1]
  uc=np.unique(cl); idxs=[np.where(cl==u)[0] for u in uc]; K=len(idxs)
  for arm in ['domain','iid']:
    rng=np.random.default_rng(2024 if arm=='domain' else 4048)
    for ia,ib in itertools.combinations(range(6),2):
      nm=SHORT[MODELS[ia]]+':'+SHORT[MODELS[ib]]
      g=(A[ib]-A[ia]) if reg=='acc' else (A[ia]-A[ib]); g=np.asarray(g,float)
      sg=-1.0 if g.mean()>0 else 1.0; h=sg*g
      for rep in range(NS):
        if arm=='domain':
          pm=rng.permutation(K); sa=np.concatenate([idxs[j] for j in pm[:K//2]]); sb=np.concatenate([idxs[j] for j in pm[K//2:]])
        else:
          pm=rng.permutation(n); sa=pm[:n//2]; sb=pm[n//2:]
        hA=h[sa]; hB=h[sb]; rA,bA=rho(hA)
        fin=bool(np.isfinite(rA))
        se=seif(hA,bA) if (fin and rA>0) else 0.0
        cert=bool((len(hB)*rA)>=LGD) if fin else True
        SP.append({'reg':reg,'arm':arm,'pair':nm,'rep':rep,'rho':(float(rA) if fin else None),
          'se':se,'nB':int(len(hB)),'cert':cert,'rev':bool(hB.mean()>=0),'fin':fin})
      lg('split %s %s %s done %.0fs'%(reg,arm,nm,time.time()-T0))
json.dump({'rows':SP,'NS':NS},open('out/run18b.json','w'))
lg('SPLITS done rows=%d %.0fs'%(len(SP),time.time()-T0))
KAP=[0.0,0.5,1.0,1.5,2.0,2.5,3.0,4.0,5.0,6.0,8.0,10.0]
def evalk(rows,k):
  nc=0; nv=0
  for r in rows:
    if not r['cert']: continue
    if r['fin'] and r['rho'] is not None and r['rho']<k*r['se']: continue
    nc+=1
    if r['rev']: nv+=1
  return nc,nv
GAT=[]
for reg in ['raw','acc']:
  for arm in ['domain','iid']:
    S=[r for r in SP if r['reg']==reg and r['arm']==arm]
    prs=sorted(set(r['pair'] for r in S)); byp={p:[r for r in S if r['pair']==p] for p in prs}
    nc0,nv0=evalk(S,0.0)
    lg('GATE %s %s baseline certified=%d violations=%d rate=%.6f'%(reg,arm,nc0,nv0,nv0/max(nc0,1)))
    for k in KAP:
      nc,nv=evalk(S,k)
      lg('  GATE %s %s kappa=%.1f certified=%d violations=%d rate=%.6f retention=%.4f'%(reg,arm,k,nc,nv,nv/max(nc,1),nc/max(nc0,1)))
    rng=np.random.default_rng(77); HV=[]; HC=[]; HK=[]
    for tr in range(200):
      pp=list(rng.permutation(prs))
      for a,b in [(pp[:7],pp[7:]),(pp[7:],pp[:7])]:
        TR=[r for p in a for r in byp[p]]; TE=[r for p in b for r in byp[p]]
        ks=None
        for k in KAP:
          c,v=evalk(TR,k)
          if v==0: ks=k; break
        if ks is None: ks=KAP[-1]
        c,v=evalk(TE,ks); c0,v0=evalk(TE,0.0)
        HK.append(ks); HV.append(v); HC.append(c/max(c0,1))
    lg('GATE %s %s HELDOUT kappa med=%.2f mean=%.3f | viol mean=%.4f max=%d zero_frac=%.4f | retention mean=%.4f'%(
      reg,arm,float(np.median(HK)),float(np.mean(HK)),float(np.mean(HV)),int(np.max(HV)),
      float(np.mean(np.array(HV)==0)),float(np.mean(HC))))
    GAT.append({'reg':reg,'arm':arm,'base_cert':nc0,'base_viol':nv0,'kappa_med':float(np.median(HK)),
      'held_viol_mean':float(np.mean(HV)),'held_viol_max':int(np.max(HV)),
      'held_zero_frac':float(np.mean(np.array(HV)==0)),'retention_mean':float(np.mean(HC)),
      'curve':[{'k':k,'cert':evalk(S,k)[0],'viol':evalk(S,k)[1]} for k in KAP]})
json.dump({'gate':GAT},open('out/run18c.json','w'))
lg('ALL DONE %.0fs'%(time.time()-T0))
