import sys,json,itertools,numpy as np
sys.path.insert(0,'../src')
import hfio,loader,ccb
from loader import MODELS,SHORT,best_file
from loader2 import rows
from scipy.stats import spearmanr
rng=np.random.default_rng(20260726)

def load_rowaligned(taskpat,mode,want=None):
    D={};files={};nr={}
    for m in MODELS:
        p,t=best_file(m,taskpat,want)
        k,v=rows(t,mode)
        D[m]=v;files[m]=p;nr[m]=len(v)
        # also keep keys of first model for order check
        if m==MODELS[0]: base=k
        else:
            assert len(k)==len(base), (m,len(k),len(base))
            assert all(a==b for a,b in zip(k,base)), 'row order mismatch '+m
    return D,files,nr

def randflip(g,rho,ndraw,rng,iters=45):
    n=g.size; hits=0; best=-1e300; vals=np.empty(ndraw)
    for d in range(ndraw):
        z=rng.standard_normal(n); z-=z.mean()
        lo,hi=0.0,1.0
        def kl(s):
            e=s*z; e-=e.max(); w=np.exp(e); w/=w.sum()
            return float(w@np.log(np.maximum(w*n,1e-300))),w
        while kl(hi)[0]<rho and hi<1e8: hi*=2.0
        for _ in range(iters):
            mid=0.5*(lo+hi)
            if kl(mid)[0]<rho: lo=mid
            else: hi=mid
        _k,w=kl(0.5*(lo+hi))
        v=float(w@g); vals[d]=v
        if v>=0: hits+=1
        best=max(best,v)
    return hits,best,vals

OUT={}
for task,pat,want in [('arc',r'arc.challenge',1172),('hellaswag',r'hellaswag',10042)]:
    for mode in ['raw','softmax','lennorm']:
        try:
            D,files,nr=load_rowaligned(pat,mode,want)
        except Exception as e:
            OUT[task+'/'+mode]={'error':str(e)[:200]}; continue
        n=len(next(iter(D.values())))
        recs=[]
        for a,b in itertools.combinations(MODELS,2):
            la,lb=D[a],D[b]
            # orient so nominal winner is the one with lower mean loss
            if la.mean()<=lb.mean(): W,Lo=a,b
            else: W,Lo=b,a
            g=D[W]-D[Lo]   # mean < 0 : winner better
            r=ccb.ccb(g)
            rec={'task':task,'mode':mode,'winner':SHORT[W],'loser':SHORT[Lo],
                 'n':n,'m':-r['mean'],'sd':r['sd']}
            if np.isfinite(r.get('rho',np.nan)):
                vm,bvm,v0,vb=ccb.vmin(g,r['beta'])
                w=r['w']; o=np.argsort(-g)
                cw=np.cumsum(w[o])
                rec.update(rho=r['rho'],half_t2=r['half_t2'],tau=r['tau'],skew=r['skew'],
                    beta=r['beta'],Kdd=r['Kdd'],Kddd=r['Kddd'],EQg=r['EQg'],kl_check=r['kl_check'],
                    vmin=vm,vmin_at=bvm,v_at0=v0,v_at_beta=vb,
                    mass_top1=float(cw[max(int(0.01*n)-1,0)]),
                    mass_top5=float(cw[max(int(0.05*n)-1,0)]),
                    mass_top10=float(cw[max(int(0.10*n)-1,0)]),
                    spearman_w_g=float(spearmanr(w,g).correlation))
                if mode=='raw':
                    for mult in [1,2,3,5]:
                        h,bst,_v=randflip(g,mult*r['rho'],400,rng)
                        rec['rand%dx_flips'%mult]=h
                        rec['rand%dx_maxEQg'%mult]=bst
            else:
                rec.update(rho=None)
            recs.append(rec)
        OUT[task+'/'+mode]={'files':{SHORT[m]:files[m] for m in files},'n_raw':{SHORT[m]:nr[m] for m in nr},'pairs':recs}
        print('done',task,mode,len(recs),flush=True)

with open('out/run1.json','w') as f:
    json.dump(OUT,f,indent=1,default=float)
print('WROTE run1.json')
