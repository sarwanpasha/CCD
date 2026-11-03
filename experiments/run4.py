import sys,json,itertools,numpy as np
sys.path.insert(0,'../src')
import hfio,loader,ccb
from loader import MODELS,SHORT,best_file,REPO
from loader2 import rows
TASKS=[('arc',r'arc.challenge',1172),('hellaswag',r'hellaswag',10042)]
OUT={'provenance':{},'pairs':[]}
for m in MODELS:
    try:
        d=json.loads(hfio._get('https://huggingface.co/api/datasets/'+REPO+m).decode())
        OUT['provenance'][SHORT[m]]={'repo':REPO+m,'sha':d.get('sha'),'lastModified':d.get('lastModified'),'downloads':d.get('downloads')}
    except Exception as e:
        OUT['provenance'][SHORT[m]]={'repo':REPO+m,'error':str(e)[:180]}
print('prov done',flush=True)
for task,pat,want in TASKS:
    T={}
    for m in MODELS:
        p,t=best_file(m,pat,want); T[m]=t
        OUT.setdefault('files',{}).setdefault(task,{})[SHORT[m]]=p
    for mode in ['raw','softmax','lennorm']:
        D={}
        for m in MODELS:
            k,v=rows(T[m],mode); D[m]=np.asarray(v,dtype=float)
        for a,b in itertools.combinations(MODELS,2):
            W,Lo=(a,b) if D[a].mean()<=D[b].mean() else (b,a)
            g=D[W]-D[Lo]
            r=ccb.ccb(g)
            gmx=float(g.max()); gmn=float(g.min()); B=float(np.max(np.abs(g)))
            mu=float(g.mean()); sd=float(g.std(ddof=1))
            rec=dict(task=task,mode=mode,winner=SHORT[W],loser=SHORT[Lo],n=int(len(g)),
              mean=mu,sd=sd,B=B,gmax=gmx,gmin=gmn,rng=gmx-gmn,
              p_pos=float((g>0).mean()),p_zero=float((g==0).mean()),p_neg=float((g<0).mean()),
              n_distinct=int(np.unique(np.round(g,12)).size),
              q99=float(np.quantile(np.abs(g),0.99)),q999=float(np.quantile(np.abs(g),0.999)),
              rho=r.get('rho'),beta=r.get('beta'),half_t2=r.get('half_t2'),
              A1b_mean_neg=bool(mu<0),A1b_p_pos=bool((g>0).mean()>0),
              A1b_both_signs=bool((g>0).any() and (g<0).any()),
              A1b_beta_pos_finite=bool(np.isfinite(r.get('beta',float('nan'))) and r.get('beta',0)>0),
              A1b_nondegen=bool(sd>0))
            rec['B_over_sd']=B/sd if sd>0 else None
            rec['hoeff_rho']=2*mu*mu/((gmx-gmn)**2) if gmx>gmn else None
            rec['hoeff_over_rho']=(rec['hoeff_rho']/r['rho']) if (r.get('rho') and rec['hoeff_rho']) else None
            rec['bernstein_rho']=(mu*mu/(2*(sd*sd+B*abs(mu)/3))) if sd>0 else None
            rec['bern_over_rho']=(rec['bernstein_rho']/r['rho']) if r.get('rho') else None
            OUT['pairs'].append(rec)
        print('done',task,mode,flush=True)
json.dump(OUT,open('out/run4.json','w'),indent=1,default=float)
print('WROTE run4.json',len(OUT['pairs']))
