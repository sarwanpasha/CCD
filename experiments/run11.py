exec(open('run9b.py').read().split("_TC={}")[0])
import numpy as np, json, time, itertools, re
from scipy.optimize import minimize
_TC={}; _orig=hfio.tree
def _tree(repo):
    if repo not in _TC: _TC[repo]=_orig(repo)
    return _TC[repo]
hfio.tree=_tree
t0=time.time()
paths=[p for p,s in _tree(REPO+MODELS[0])]
SUB=sorted(set(re.search(r'hendrycksTest-(.+?)\|5',p).group(1) for p in paths
    if 'hendrycksTest-' in p and p.endswith('.parquet')))
LOSS={}; SUBID=[]; keep=[]
for s in SUB:
    try: r=load_mmlu(s)
    except Exception: continue
    if r is None: continue
    L,n=r
    if n<20: continue
    keep.append(s)
    for m in MODELS: LOSS.setdefault(m,[]).append(L[m])
    SUBID.append(np.full(n,len(keep)-1))
X={m:np.concatenate(LOSS[m]) for m in MODELS}
SID=np.concatenate(SUBID); NS=int(SID.max())+1; N=len(SID)
print('MMLU subjects=%d items=%d %.1fs'%(NS,N,time.time()-t0),flush=True)
PHI=[0.0,0.25,0.5,0.75,0.9,1.0]; R=250
rng=np.random.default_rng(20260728); recs=[]
for a,b in itertools.combinations(MODELS,2):
    g0=X[a]-X[b]; nm=SHORT[a]+':'+SHORT[b]
    for phi in PHI:
        for i in range(R):
            perm=rng.permutation(NS); ss=np.isin(SID,perm[:NS//2])
            km=rng.random(N)<phi; ii=rng.random(N)<0.5
            side=np.where(km,ss,ii)
            gA=g0[side]; gB=g0[~side]
            if len(gA)<50 or len(gB)<50: continue
            sg=-1.0 if gA.mean()>0 else 1.0
            hA=sg*gA; hB=sg*gB
            rv,bs=rho(hA); mm=hA.mean(); sd=hA.std(ddof=1)
            recs.append({'p':nm,'phi':phi,'rho':(None if not np.isfinite(rv) else float(rv)),
              'nA':int(len(hA)),'nB':int(len(hB)),'mA':float(mm),'sdA':float(sd),
              'mB':float(hB.mean()),'shift':float((hB.mean()-mm)/sd) if sd>0 else 0.0,
              't':float(mm/(sd/np.sqrt(len(hA)))) if sd>0 else 0.0,'rev':int(hB.mean()>=0)})
    print('pair %s %.1fs recs=%d'%(nm,time.time()-t0,len(recs)),flush=True)
json.dump({'NS':int(NS),'N':int(N),'PHI':PHI,'R':R,'recs':recs},open('out/run11.json','w'))
print('WROTE out/run11.json recs=%d'%len(recs),flush=True)

from loader2 import load_task
D=load_task('hellaswag',mode='raw'); L=D['L']; ms=D['models']
tgt=ms[0]; riv=[m for m in ms if m!=tgt]
G=np.stack([L[tgt]-L[m] for m in riv],axis=1)
n,k=G.shape; print('kway G shape',G.shape,flush=True)
def Lam(lam):
    z=G@lam; mz=z.max(); e=np.exp(z-mz); v=mz+np.log(e.mean())
    return v,(G*e[:,None]).sum(0)/e.sum()
TM=[]
for rep in range(20):
    t1=time.time()
    res=minimize(Lam,np.zeros(k),jac=True,method='L-BFGS-B',bounds=[(0,None)]*k,
                 options={'ftol':1e-15,'gtol':1e-13,'maxiter':500})
    TM.append(time.time()-t1)
TM=np.array(TM)
print('kway target=%s k=%d n=%d rho_S=%.8f nit=%d nfev=%d'%(SHORT[tgt],k,n,-res.fun,res.nit,res.nfev),flush=True)
print('kway lambda=%s'%np.array2string(res.x,precision=6),flush=True)
print('kway time 20 reps: min %.4fs med %.4fs max %.4fs'%(TM.min(),np.median(TM),TM.max()),flush=True)
print('kway active=%s'%[SHORT[riv[j]] for j in range(k) if res.x[j]>1e-9],flush=True)
