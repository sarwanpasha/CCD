import sys,json,itertools,numpy as np
sys.path.insert(0,'../src')
import ccb
from loader import MODELS,SHORT,best_file
from loader2 import rows
from scipy.stats import spearmanr
from scipy.special import logsumexp
from scipy.optimize import minimize
rng=np.random.default_rng(20260726)
TASKS=[('arc',r'arc.challenge',1172),('hellaswag',r'hellaswag',10042)]

def load(pat,mode,want):
    D={};F={};base=None
    for m in MODELS:
        p,t=best_file(m,pat,want)
        k,v=rows(t,mode); D[m]=v; F[m]=p
        if base is None: base=k
        else: assert all(a==b for a,b in zip(k,base)),'order '+m
    return D,F

def inter_rho(G):
    n,k=G.shape
    def fg(lam):
        z=G@lam; L=logsumexp(z)-np.log(n)
        w=np.exp(z-logsumexp(z))
        return L, G.T@w
    r=minimize(lambda l:fg(l)[0],np.zeros(k),jac=lambda l:fg(l)[1],
               bounds=[(0,None)]*k,method='L-BFGS-B',
               options={'ftol':1e-18,'gtol':1e-14,'maxiter':20000})
    lam=np.maximum(r.x,0.0); z=G@lam; w=np.exp(z-logsumexp(z))
    return float(-r.fun),lam,w,(G.T@w)

OUT={}
for task,pat,want in TASKS:
    D,F=load(pat,'raw',want)
    n=len(next(iter(D.values())))
    means={m:float(D[m].mean()) for m in MODELS}
    order=sorted(MODELS,key=lambda m:means[m])
    res={'n':n,'files':{SHORT[m]:F[m] for m in MODELS},
         'mean_loss':{SHORT[m]:means[m] for m in MODELS},
         'ranking_best_first':[SHORT[m] for m in order],'targets':[]}
    for tgt in MODELS:
        rivals=[m for m in MODELS if means[m]>means[tgt]]
        if len(rivals)<2: continue
        gj={m:(D[tgt]-D[m]) for m in rivals}   # E[gj]<0 ; gj>=0 means rival m overtakes tgt
        pw={}
        for m in rivals:
            c=ccb.ccb(gj[m])
            pw[SHORT[m]]={'rho':c.get('rho'),'m':-c['mean'],'skew':c.get('skew'),'beta':c.get('beta')}
        # correlation structure of the g_j
        Gall=np.column_stack([gj[m] for m in rivals])
        C=np.corrcoef(Gall.T)
        ent={'target':SHORT[tgt],'rivals':[SHORT[m] for m in rivals],
             'pairwise':pw,'corr':C.tolist(),'subsets':{}}
        for r_ in [2,3]:
            if len(rivals)<r_: continue
            best=None; rows_=[]
            for S in itertools.combinations(rivals,r_):
                G=np.column_stack([gj[m] for m in S])
                rho,lam,w,eq=inter_rho(G)
                a=int((lam>1e-10).sum())
                sp=[float(spearmanr(w,G[:,j]).correlation) for j in range(r_)]
                rec={'S':[SHORT[m] for m in S],'rho':rho,'lam':lam.tolist(),'active':a,
                     'EQg':eq.tolist(),'spearman_w_gj':sp,
                     'sum_pairwise':sum(pw[SHORT[m]]['rho'] for m in S),
                     'max_pairwise':max(pw[SHORT[m]]['rho'] for m in S)}
                rows_.append(rec)
                if best is None or rho<best['rho']: best=rec
            cheap=sorted(rivals,key=lambda m:pw[SHORT[m]]['rho'])[:r_]
            ent['subsets'][str(r_)]={'all':rows_,'argmin':best['S'],'rho_r':best['rho'],
                'naive_r_smallest':[SHORT[m] for m in cheap],
                'argmin_is_naive':sorted(best['S'])==sorted(SHORT[m] for m in cheap),
                'active_at_argmin':best['active'],
                'n_active_hist':{str(k):sum(1 for x in rows_ if x['active']==k) for k in range(0,r_+1)}}
        res['targets'].append(ent)
    OUT[task]=res
    print('done',task,flush=True)
with open('out/run2.json','w') as f: json.dump(OUT,f,indent=1,default=float)
print('WROTE run2.json')
