import numpy as np, json, re, time, itertools, sys, hashlib
sys.path.insert(0,'../src')
import hfio
from scipy.stats import rankdata
from loader import MODELS,SHORT,REPO,files_for

def rho(g):
    g=np.asarray(g,float); n=len(g); mg=g.mean()
    if mg>=0: return 0.0,0.0
    if g.max()<=0: return np.inf,np.inf
    sc=max(1.0,np.abs(g).mean())
    def Kp(b):
        z=b*g; z=z-z.max(); e=np.exp(z); return (g*e).sum()/e.sum()
    lo=0.0; hi=1.0; it=0
    while Kp(hi)<0:
        hi*=2.0; it+=1
        if it>300: return np.inf,np.inf
    b=0.5*hi
    for _ in range(90):
        z=b*g; z=z-z.max(); e=np.exp(z); se=e.sum()
        m1=(g*e).sum()/se
        if abs(m1)<1e-14*sc: break
        if m1>0: hi=b
        else: lo=b
        m2=(g*g*e).sum()/se-m1*m1
        nb=b-m1/m2 if m2>1e-300 else 0.5*(lo+hi)
        b=nb if lo<nb<hi else 0.5*(lo+hi)
        if hi-lo<1e-15*max(1.0,hi): break
    z=b*g; mz=z.max(); e=np.exp(z-mz)
    return -(mz+np.log(e.sum()/n)),b

def auc(score,lab):
    s=np.asarray(score,float); l=np.asarray(lab).astype(int)
    a=int((l==1).sum()); b=int((l==0).sum())
    if a==0 or b==0: return float('nan')
    r=rankdata(s)
    return float((r[l==1].sum()-a*(a+1)/2.0)/(a*b))

def ekey(s): return hashlib.md5(str(s).encode()).hexdigest()

def load_mmlu(subj):
    D={}
    for m in MODELS:
        c=files_for(m,r'hendrycksTest-'+re.escape(subj)+r'\|5')
        if not c: return None
        best=None
        for p,sz in c:
            try: t=hfio.table(REPO+m,p)
            except Exception: continue
            if best is None or t.num_rows>best.num_rows: best=t
        if best is None: return None
        d=best.to_pydict(); ex=d['example']; gold=d['gold']; pred=d['predictions']
        K={}
        for i in range(len(ex)):
            gi=gold[i]
            if isinstance(gi,(list,tuple)): gi=gi[0]
            gi=int(gi); pr=np.asarray(pred[i],dtype=np.float64)
            K[ekey(ex[i])]=float(-pr[gi])
        D[m]=K
    common=sorted(set.intersection(*[set(D[m]) for m in MODELS]))
    if not common: return None
    return {m:np.array([D[m][k] for k in common]) for m in MODELS},len(common)

_TC={}; _orig=hfio.tree
def _tree(repo):
    if repo not in _TC: _TC[repo]=_orig(repo)
    return _TC[repo]
hfio.tree=_tree

paths=[p for p,s in _tree(REPO+MODELS[0])]
SUB=sorted(set(re.search(r'hendrycksTest-(.+?)\|5',p).group(1) for p in paths
    if 'hendrycksTest-' in p and p.endswith('.parquet')))
print('subjects in tree:',len(SUB),flush=True)
LOSS={}; SUBID=[]; keep=[]; t0=time.time()
for si,s in enumerate(SUB):
    try: r=load_mmlu(s)
    except Exception as e:
        print('SKIPERR',s,repr(e)[:70],flush=True); continue
    if r is None: print('SKIPNONE',s,flush=True); continue
    L,n=r
    if n<20: print('SKIPN',s,n,flush=True); continue
    keep.append(s)
    for m in MODELS: LOSS.setdefault(m,[]).append(L[m])
    SUBID.append(np.full(n,len(keep)-1))
    print('%d/%d %s n=%d cum=%.1fs'%(si+1,len(SUB),s,n,time.time()-t0),flush=True)
X={m:np.concatenate(LOSS[m]) for m in MODELS}
SID=np.concatenate(SUBID); NS=int(SID.max())+1; N=len(SID)
print('KEPT subjects=%d items=%d'%(NS,N),flush=True)
R=400; rng=np.random.default_rng(20260727); recs=[]
for a,b in itertools.combinations(MODELS,2):
    g0=X[a]-X[b]; nm=SHORT[a]+':'+SHORT[b]
    rf,bf=rho(g0 if g0.mean()<0 else -g0)
    print('PAIR %s mean=%.6f rho_full=%.6f'%(nm,g0.mean(),rf),flush=True)
    for arm in ('domain','iid'):
        for i in range(R):
            if arm=='domain':
                perm=rng.permutation(NS); mA=np.isin(SID,perm[:NS//2])
            else:
                mA=rng.random(N)<0.5
            gA=g0[mA]; gB=g0[~mA]
            if len(gA)<50 or len(gB)<50: continue
            sg=-1.0 if gA.mean()>0 else 1.0
            hA=sg*gA; hB=sg*gB
            rv,bs=rho(hA); mm=hA.mean(); sd=hA.std(ddof=1)
            tt=mm/(sd/np.sqrt(len(hA))) if sd>0 else 0.0
            recs.append({'p':nm,'arm':arm,'i':i,'sg':sg,
              'rho':(None if not np.isfinite(rv) else float(rv)),'beta':float(bs) if np.isfinite(bs) else None,
              'nA':int(len(hA)),'nB':int(len(hB)),'nsA':int(mA.sum()>0 and len(set(SID[mA].tolist()))),
              'mA':float(mm),'sdA':float(sd),'t':float(tt),'mB':float(hB.mean()),
              'rev':int(hB.mean()>=0)})
    print('  %s done %.1fs recs=%d'%(nm,time.time()-t0,len(recs)),flush=True)
json.dump({'subjects':keep,'models':[SHORT[m] for m in MODELS],'N':int(N),'NS':int(NS),
    'R':R,'recs':recs},open('out/run9.json','w'))
print('WROTE out/run9.json recs=%d %.1fs'%(len(recs),time.time()-t0),flush=True)
