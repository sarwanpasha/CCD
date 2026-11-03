import sys,json,itertools,time,numpy as np
sys.path.insert(0,'../src')
import hfio
from loader import MODELS,SHORT,best_file
from loader2 import ckey
from scipy.stats import skew,kurtosis
from scipy.optimize import minimize

def rho(g):
    g=np.asarray(g,float); n=len(g); mu=g.mean()
    if mu>=0: return 0.0,0.0
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

def boot(g,bstar,B=4000,seed=7,blk=250,G=800):
    n=len(g)
    bg=np.concatenate([[0.0],np.geomspace(max(bstar/60.0,1e-8),bstar*25.0,G-1)])
    Z=np.outer(bg,g); M=Z.max(axis=1); E=np.exp(Z-M[:,None]); del Z
    rng=np.random.default_rng(seed); out=[]
    for s in range(0,B,blk):
        k=min(blk,B-s)
        C=rng.multinomial(n,np.full(n,1.0/n),size=k).astype(np.float64)
        A=E@C.T
        Kb=M[:,None]+np.log(np.maximum(A,1e-300)/n)
        out.append(-Kb.min(axis=0))
    v=np.concatenate(out)
    return v

def fsbound(g,bstar,delta=0.05,G=400):
    n=len(g); B=float(np.abs(g).max())
    bg=np.linspace(0.0,max(2.0*bstar,1e-9),G)
    lg=np.log(2.0*G/delta)
    Z=np.outer(bg,g); M=Z.max(axis=1)
    EX=np.exp(M)*np.exp(Z-M[:,None]).mean(axis=1)
    rngsp=(np.exp(bg*B)-np.exp(-bg*B))
    dev=rngsp*np.sqrt(lg/(2.0*n))
    eps=dev/np.maximum(EX,1e-300)
    dK=np.where(eps<1.0,-np.log(np.maximum(1.0-eps,1e-300)),np.inf)
    A=float(np.max(rngsp/np.maximum(EX,1e-300)))
    return float(np.max(dK)),B,A

def cumexp(g):
    mu=float(g.mean()); sd=float(g.std(ddof=0)); s3=float(skew(g)); k4=float(kurtosis(g))
    k3=s3*sd**3; kk4=k4*sd**4
    r2=mu*mu/(2*sd*sd)
    r3=r2+k3*mu**3/(6*sd**6)
    f=lambda b: mu*b+sd*sd*b*b/2.0+k3*b**3/6.0+kk4*b**4/24.0
    b0=-mu/(sd*sd)
    gr=np.linspace(1e-9,4*b0,20000); vals=f(gr); r4=float(-vals.min())
    return mu,sd,s3,k4,r2,r3,r4,b0

TASKS=[('arc',r'arc.challenge',1172),('hellaswag',r'hellaswag',10042)]
CONV=['raw','chr','tokbar']
OUT={'cases':[],'inf':{c:0 for c in CONV}}
t0=time.time()
for task,pat,want in TASKS:
    R={}
    for m in MODELS:
        p,t=best_file(m,pat,want); d=t.to_pydict()
        ch=d['choices']; gold=d['gold']; pred=d['predictions']; ct=d['cont_tokens']
        K={}
        for i in range(len(ch)):
            gi=gold[i]
            if isinstance(gi,(list,tuple)): gi=gi[0]
            gi=int(gi); pr=np.asarray(pred[i],dtype=np.float64)
            tok=len(ct[i][gi]) if ct is not None else -1
            chr_=len(str(ch[i][gi]))
            K[ckey(ch[i],gi)]=(float(-pr[gi]),max(tok,1),max(chr_,1))
        R[m]=K
        print('load',task,SHORT[m],len(K),'%.1fs'%(time.time()-t0),flush=True)
    common=sorted(set.intersection(*[set(R[m]) for m in MODELS]))
    A={m:np.array([R[m][k][0] for k in common]) for m in MODELS}
    TOK={m:np.array([R[m][k][1] for k in common],dtype=float) for m in MODELS}
    CH=np.array([R[MODELS[0]][k][2] for k in common],dtype=float)
    TBAR=np.mean([TOK[m] for m in MODELS],axis=0)
    DEN={'raw':np.ones(len(common)),'chr':CH,'tokbar':TBAR}
    print('TASK',task,'n=',len(common),flush=True)
    for cv in CONV:
        L={m:A[m]/DEN[cv] for m in MODELS}
        for a,b in itertools.combinations(MODELS,2):
            g=L[a]-L[b]; nm=SHORT[a]+':'+SHORT[b]
            sg=-1.0 if g.mean()>0 else 1.0; h=sg*g
            win=SHORT[a] if sg>0 else SHORT[b]
            rh,bs=rho(h)
            rec={'task':task,'conv':cv,'pair':nm,'winner':win,'n':len(h)}
            if not np.isfinite(rh):
                OUT['inf'][cv]+=1
                rec.update({'rho':None,'note':'INF max_g=%.6g'%float(h.max())})
                OUT['cases'].append(rec); print('INF',task,cv,nm,flush=True); continue
            v=boot(h,bs)
            mu,sd,s3,k4,r2,r3,r4,b0=cumexp(h)
            cfs,Bmax,Acon=fsbound(h,bs)
            nreq=Acon*Acon*np.log(2*400/0.05)/(2.0*(1.0-np.exp(-rh))**2)
            rec.update({'rho':float(rh),'beta':float(bs),'ci':[float(np.percentile(v,2.5)),float(np.percentile(v,97.5))],
              'boot_sd':float(v.std(ddof=1)),'boot_mean':float(v.mean()),
              'mu':mu,'sd':sd,'skew':s3,'exkurt':k4,'s':float(-mu/sd),'t':float(-mu/(sd/np.sqrt(len(h)))),
              'r2':r2,'r3':r3,'r4':r4,'b0':float(b0),'Bmax':Bmax,'cfs':cfs,'nreq':float(nreq)})
            OUT['cases'].append(rec)
        print('conv',task,cv,'%.1fs'%(time.time()-t0),flush=True)
json.dump(OUT,open('out/run10.json','w'))
print('WROTE out/run10.json cases=%d %.1fs'%(len(OUT['cases']),time.time()-t0),flush=True)
