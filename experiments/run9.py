import numpy as np, json, glob, re, time, itertools, sys
from scipy.stats import rankdata
from loader2 import load_task
from loader import SHORT

def rho(g):
    g=np.asarray(g,float); n=len(g); mg=g.mean()
    if mg>=0: return 0.0
    if g.max()<=0: return np.inf
    sc=max(1.0,np.abs(g).mean())
    def Kp(b):
        z=b*g; z=z-z.max(); e=np.exp(z); return (g*e).sum()/e.sum()
    lo=0.0; hi=1.0; it=0
    while Kp(hi)<0:
        hi*=2.0; it+=1
        if it>200: return np.inf
    b=0.5*hi
    for _ in range(80):
        z=b*g; z=z-z.max(); e=np.exp(z); se=e.sum()
        m1=(g*e).sum()/se
        if abs(m1)<1e-13*sc: break
        if m1>0: hi=b
        else: lo=b
        m2=(g*g*e).sum()/se-m1*m1
        nb=b-m1/m2 if m2>1e-300 else 0.5*(lo+hi)
        b=nb if lo<nb<hi else 0.5*(lo+hi)
        if hi-lo<1e-14*max(1.0,hi): break
    z=b*g; mz=z.max(); e=np.exp(z-mz)
    return -(mz+np.log(e.sum()/n)), b

def rhoq(g):
    r=rho(g)
    return r if isinstance(r,float) else r[0]

def auc(score,lab):
    s=np.asarray(score,float); l=np.asarray(lab).astype(int)
    np_=int((l==1).sum()); nq=int((l==0).sum())
    if np_==0 or nq==0: return float('nan')
    r=rankdata(s)
    return float((r[l==1].sum()-np_*(np_+1)/2.0)/(np_*nq))

SUB=sorted(set(re.search(r'harness_hendrycksTest-(.+?)_5_2\d',p).group(1)
    for p in glob.glob('data/*hendrycksTest*.parquet')
    if re.search(r'harness_hendrycksTest-(.+?)_5_2\d',p)))
print('subjects found',len(SUB),flush=True)
LOSS={}; SUBID=[]; keep=[]; t0=time.time()
for si,s in enumerate(SUB):
    try: D=load_task(re.escape('hendrycksTest-'+s+'|5'),mode='raw')
    except Exception as e:
        print('SKIPERR',s,repr(e)[:70],flush=True); continue
    if len(D['models'])<6:
        print('SKIPNM',s,len(D['models']),flush=True); continue
    L=D['L']; n=len(next(iter(L.values())))
    if n<20:
        print('SKIPN',s,n,flush=True); continue
    keep.append(s)
    for m,v in L.items(): LOSS.setdefault(m,[]).append(v)
    SUBID.append(np.full(n,len(keep)-1))
    print('%d/%d %s n=%d cum=%.1fs'%(si+1,len(SUB),s,n,time.time()-t0),flush=True)
MODS=sorted(LOSS.keys())
X={m:np.concatenate(LOSS[m]) for m in MODS}
SID=np.concatenate(SUBID); NS=int(SID.max())+1; N=len(SID)
print('KEPT subjects=%d items=%d models=%s'%(NS,N,[SHORT[m] for m in MODS]),flush=True)
R=400; rng=np.random.default_rng(20260727); recs=[]
for a,b in itertools.combinations(MODS,2):
    g0=X[a]-X[b]; nm=SHORT[a]+':'+SHORT[b]
    rfull=rho(g0 if g0.mean()<0 else -g0)
    rf=rfull if isinstance(rfull,float) else rfull[0]
    print('PAIR %s full_mean=%.6f full_rho=%.6f'%(nm,g0.mean(),rf),flush=True)
    for arm in ('domain','iid'):
        for r in range(R):
            if arm=='domain':
                perm=rng.permutation(NS); mA=np.isin(SID,perm[:NS//2])
            else:
                mA=rng.random(N)<0.5
            gA=g0[mA]; gB=g0[~mA]
            if len(gA)<50 or len(gB)<50: continue
            sg=-1.0 if gA.mean()>0 else 1.0
            hA=sg*gA; hB=sg*gB
            rr=rho(hA); rv=rr if isinstance(rr,float) else rr[0]
            bs=0.0 if isinstance(rr,float) else rr[1]
            mm=hA.mean(); sd=hA.std(ddof=1)
            tt=mm/(sd/np.sqrt(len(hA))) if sd>0 else 0.0
            recs.append({'p':nm,'arm':arm,'i':r,'sg':sg,
              'rho':(None if not np.isfinite(rv) else float(rv)),'beta':float(bs),
              'nA':int(len(hA)),'nB':int(len(hB)),'mA':float(mm),'sdA':float(sd),
              't':float(tt),'mB':float(hB.mean()),'rev':int(hB.mean()>=0)})
    print('  done %s %.1fs recs=%d'%(nm,time.time()-t0,len(recs)),flush=True)
json.dump({'subjects':keep,'models':[SHORT[m] for m in MODS],'N':int(N),'NS':int(NS),
    'R':R,'recs':recs},open('out/run9.json','w'))
print('WROTE out/run9.json recs=%d elapsed=%.1fs'%(len(recs),time.time()-t0),flush=True)
