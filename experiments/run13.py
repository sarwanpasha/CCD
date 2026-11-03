exec(open('run10.py').read().split("def fsbound")[0])
exec(open('run9b.py').read().split("def rho")[0])
import numpy as np, json, time, itertools, re, hashlib
from scipy.stats import skew,kurtosis
_TC={}; _o=hfio.tree
def _tree(r):
    if r not in _TC: _TC[r]=_o(r)
    return _TC[r]
hfio.tree=_tree
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
        d=best.to_pydict(); ex=d['example']; gold=d['gold']; pr=d['predictions']
        K={}
        for i in range(len(ex)):
            gi=gold[i]
            if isinstance(gi,(list,tuple)): gi=gi[0]
            K[ekey(ex[i])]=float(-np.asarray(pr[i],dtype=np.float64)[int(gi)])
        D[m]=K
    com=sorted(set.intersection(*[set(D[m]) for m in MODELS]))
    if not com: return None
    return {m:np.array([D[m][k] for k in com]) for m in MODELS},len(com)
def load_wino():
    D={}
    for m in MODELS:
        d=None; used=None
        for p,sz in files_for(m,r'winogrande\|5'):
            dd=hfio.table(REPO+m,p).to_pydict()
            if set(len(x) for x in dd['predictions'])=={2}:
                if d is None or len(dd['example'])>len(d['example']): d,used=dd,p
            else:
                print('  REJECT %s %s predlens=%s'%(SHORT[m],p[:24],
                  sorted(set(len(x) for x in dd['predictions']))[:4]),flush=True)
        if d is None: raise RuntimeError('no valid winogrande file for '+m)
        print('  USE %s %s'%(SHORT[m],used[:24]),flush=True)
        ex=d['example']; o1=d['option1']; o2=d['option2']; an=d['answer']; pr=d['predictions']
        K={}
        for i in range(len(ex)):
            g=int(str(an[i]).strip())-1
            K[ekey(str(ex[i])+'##'+str(o1[i])+'##'+str(o2[i]))]=float(-np.asarray(pr[i],dtype=np.float64)[g])
        D[m]=K
        print('wino',SHORT[m],len(K),flush=True)
    com=sorted(set.intersection(*[set(D[m]) for m in MODELS]))
    return {m:np.array([D[m][k] for k in com]) for m in MODELS},len(com)
t0=time.time(); TASKS={}
W,nw=load_wino(); TASKS['winogrande']=W
print('winogrande n=%d %.1fs'%(nw,time.time()-t0),flush=True)
paths=[p for p,s in _tree(REPO+MODELS[0])]
SUB=sorted(set(re.search(r'hendrycksTest-(.+?)\|5',p).group(1) for p in paths
  if 'hendrycksTest-' in p and p.endswith('.parquet')))
acc={}; nm2=0
for s in SUB:
    try: r=load_mmlu(s)
    except Exception: continue
    if r is None: continue
    L,n=r
    for m in MODELS: acc.setdefault(m,[]).append(L[m])
    nm2+=n
TASKS['mmlu']={m:np.concatenate(acc[m]) for m in MODELS}
print('mmlu pooled n=%d %.1fs'%(nm2,time.time()-t0),flush=True)
OUT=[]
for tk,L in TASKS.items():
    n=len(L[MODELS[0]])
    for a,b in itertools.combinations(MODELS,2):
        g=L[a]-L[b]; sg=-1.0 if g.mean()>0 else 1.0; h=sg*g
        win=SHORT[a] if sg>0 else SHORT[b]
        rh,bs=rho(h)
        rec={'task':tk,'n':int(n),'pair':SHORT[a]+':'+SHORT[b],'winner':win}
        if not np.isfinite(rh):
            rec['rho']=None; OUT.append(rec); print('INF',tk,rec['pair'],flush=True); continue
        v=boot(h,bs)
        mu=float(h.mean()); sd=float(h.std(ddof=0))
        rec.update({'rho':float(rh),'beta':float(bs),'ci':[float(np.percentile(v,2.5)),float(np.percentile(v,97.5))],
          'boot_sd':float(v.std(ddof=1)),'mu':mu,'sd':sd,'skew':float(skew(h)),'exkurt':float(kurtosis(h)),
          's':float(-mu/sd),'t':float(-mu/(sd/np.sqrt(n))),'r2':mu*mu/(2*sd*sd),'B':float(np.abs(h).max())})
        OUT.append(rec)
    print('task %s done %.1fs'%(tk,time.time()-t0),flush=True)
json.dump(OUT,open('out/run13.json','w'))
print('WROTE out/run13.json cases=%d'%len(OUT),flush=True)
