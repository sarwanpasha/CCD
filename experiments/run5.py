import sys,json,itertools,numpy as np
sys.path.insert(0,'../src')
import hfio,loader,ccb
from loader import MODELS,SHORT,best_file
from loader2 import rows
from scipy.stats import spearmanr
TASKS=[('arc',r'arc.challenge',1172),('hellaswag',r'hellaswag',10042)]
def amip(g):
    n=len(g); s=np.sort(g); c=np.cumsum(s); S=g.sum()
    tail=S-c            # tail[j] = sum(s[j+1:]) after dropping s[0..j]
    ok=np.nonzero(tail>=0.0)[0]
    if ok.size==0: return n, 1.0, float('inf')
    k=int(ok[0])+1
    assert (S-c[k-1])>=0 and (k==1 or (S-c[k-2])<0)
    return k, k/n, -np.log(1.0-k/n)
OUT={'pairs':[],'kway':[]}
for task,pat,want in TASKS:
    T={}
    for m in MODELS:
        p,t=best_file(m,pat,want); T[m]=t
    for mode in ['raw','softmax','lennorm']:
        D={m:np.asarray(rows(T[m],mode)[1],dtype=float) for m in MODELS}
        for a,b in itertools.combinations(MODELS,2):
            W,Lo=(a,b) if D[a].mean()<=D[b].mean() else (b,a)
            g=D[W]-D[Lo]; n=len(g)
            r=ccb.ccb(g); rho=r.get('rho')
            if rho is None or not np.isfinite(rho):
                OUT['pairs'].append(dict(task=task,mode=mode,winner=SHORT[W],loser=SHORT[Lo],n=len(g),rho=None,degenerate=True,p_pos=float((g>0).mean()),gmax=float(g.max()),mean=float(g.mean())))
                continue
            w=np.asarray(r['w'],dtype=float)
            kl=float(np.sum(w*np.log(np.maximum(w,1e-300)*n)))
            k,al,cd=amip(g)
            o=np.argsort(g)[:k]
            rec=dict(task=task,mode=mode,winner=SHORT[W],loser=SHORT[Lo],n=n,rho=rho,
                     kl_direct=kl,kl_resid=abs(kl-rho),
                     amip_k=int(k),amip_alpha=float(al),amip_cost=float(cd),
                     eff=float(rho/cd) if cd>0 else None,
                     wmass_on_dropped=float(w[o].sum()),
                     unif_mass_on_dropped=float(al),
                     mass_ratio=float(w[o].sum()/al) if al>0 else None,
                     EQg_amip=float(g[np.argsort(g)[k:]].mean()))
            OUT['pairs'].append(rec)
        print('done',task,mode,flush=True)
json.dump(OUT,open('out/run5.json','w'),indent=1,default=float)
print('WROTE run5.json',len(OUT['pairs']))
