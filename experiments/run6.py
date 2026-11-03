import sys,json,itertools,numpy as np
sys.path.insert(0,'../src')
import hfio,loader,ccb
from loader import MODELS,SHORT,best_file
from loader2 import ckey
from scipy.stats import spearmanr,pearsonr
TASKS=[('arc',r'arc.challenge',1172),('hellaswag',r'hellaswag',10042)]
OUT={}
for task,pat,want in TASKS:
    R={}
    for m in MODELS:
        p,t=best_file(m,pat,want); d=t.to_pydict()
        ch=d['choices']; gold=d['gold']; pred=d['predictions']; ct=d['cont_tokens']
        n=len(ch); K={}
        for i in range(n):
            gi=gold[i]
            if isinstance(gi,(list,tuple)): gi=gi[0]
            gi=int(gi)
            pr=np.asarray(pred[i],dtype=np.float64)
            tok=len(ct[i][gi]) if ct is not None else -1
            chr_=len(str(ch[i][gi]))
            K[ckey(ch[i],gi)]=(float(-pr[gi]),max(tok,1),max(chr_,1))
        R[m]=K
        print(task,SHORT[m],'n',len(K),flush=True)
    common=sorted(set.intersection(*[set(R[m]) for m in MODELS]))
    A={m:np.array([R[m][k][0] for k in common]) for m in MODELS}
    TOK={m:np.array([R[m][k][1] for k in common],dtype=float) for m in MODELS}
    CH=np.array([R[MODELS[0]][k][2] for k in common],dtype=float)
    o={'n':len(common),'chars':{'mean':float(CH.mean()),'median':float(np.median(CH)),'min':float(CH.min()),'max':float(CH.max())},'tok':{},'tokcorr':{},'pairs':[]}
    for m in MODELS:
        L=TOK[m]; o['tok'][SHORT[m]]={'mean':float(L.mean()),'median':float(np.median(L)),'min':float(L.min()),'max':float(L.max()),'sum':float(L.sum())}
    for a,b in itertools.combinations(MODELS,2):
        o['tokcorr'][SHORT[a]+':'+SHORT[b]]={'spearman':float(spearmanr(TOK[a],TOK[b]).correlation),'meanratio':float((TOK[a]/TOK[b]).mean())}
    TBAR=np.mean([TOK[m] for m in MODELS],axis=0)
    for a,b in itertools.combinations(MODELS,2):
        W,Lo=(a,b) if A[a].mean()<=A[b].mean() else (b,a)
        g=A[W]-A[Lo]
        rec={'winner':SHORT[W],'loser':SHORT[Lo],'mean_g':float(g.mean()),
             'corr_g_tokbar_p':float(pearsonr(g,TBAR)[0]),'corr_g_tokbar_s':float(spearmanr(g,TBAR).correlation),
             'corr_g_chars_p':float(pearsonr(g,CH)[0]),'corr_g_chars_s':float(spearmanr(g,CH).correlation),
             'rho_raw':ccb.ccb(g).get('rho')}
        # model-independent normalisations
        for nm,DEN in [('chr',CH),('tokbar',TBAR)]:
            aa=A[a]/DEN; bb=A[b]/DEN
            W2,L2=(a,b) if aa.mean()<=bb.mean() else (b,a)
            g2=(A[W2]/DEN)-(A[L2]/DEN)
            r2=ccb.ccb(g2)
            rec['win_'+nm]=SHORT[W2]; rec['rho_'+nm]=r2.get('rho'); rec['mean_'+nm]=float(g2.mean())
            rec['flip_'+nm]=bool(SHORT[W2]!=SHORT[W])
        # per-model-token normalisation (the convention actually used)
        aa=A[a]/TOK[a]; bb=A[b]/TOK[b]
        W3,L3=(a,b) if aa.mean()<=bb.mean() else (b,a)
        g3=(A[W3]/TOK[W3])-(A[L3]/TOK[L3])
        r3=ccb.ccb(g3)
        rec['win_owntok']=SHORT[W3]; rec['rho_owntok']=r3.get('rho'); rec['flip_owntok']=bool(SHORT[W3]!=SHORT[W])
        o['pairs'].append(rec)
    OUT[task]=o
    print('done',task,flush=True)
json.dump(OUT,open('out/run6.json','w'),indent=1,default=float)
print('WROTE run6.json')
