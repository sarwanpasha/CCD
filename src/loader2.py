import os
import re,numpy as np,hfio,hashlib
from loader import MODELS,SHORT,REPO,best_file

def ckey(ch,gold):
    s='||'.join(str(c) for c in ch)+'##'+str(gold)
    return hashlib.md5(s.encode()).hexdigest()

def rows(tab,mode='raw'):
    d=tab.to_pydict()
    ch=d['choices']; gold=d['gold']; pred=d['predictions']; ct=d.get('cont_tokens')
    n=len(ch); keys=[]; out=np.empty(n)
    for i in range(n):
        pr=np.asarray(pred[i],dtype=np.float64)
        gi=gold[i]
        if isinstance(gi,(list,tuple)): gi=gi[0]
        gi=int(gi)
        if mode=='raw': v=-pr[gi]
        elif mode=='softmax':
            mx=pr.max(); lse=mx+np.log(np.exp(pr-mx).sum()); v=-(pr[gi]-lse)
        elif mode=='lennorm':
            L=np.array([max(len(c),1) for c in ct[i]],dtype=np.float64)
            v=-(pr/L)[gi]
        out[i]=v; keys.append(ckey(ch[i],gi))
    return keys,out

def load_task(taskpat,mode='raw',want_rows=None,models=None,verbose=False):
    models=models or MODELS
    D={}
    for m in models:
        p,t=best_file(m,taskpat,want_rows)
        if t is None: continue
        k,v=rows(t,mode)
        D[m]={'file':p,'keys':k,'loss':v,'n':len(k)}
    common=None
    for m in D:
        s=set(D[m]['keys'])
        common=s if common is None else (common & s)
    common=sorted(common)
    ok=all(len(set(D[m]['keys']))==D[m]['n'] for m in D)
    idx={m:{k:i for i,k in enumerate(D[m]['keys'])} for m in D}
    L={m:np.array([D[m]['loss'][idx[m][k]] for k in common]) for m in D}
    # row-order agreement diagnostic
    ms=list(D.keys()); agree=None
    if len(ms)>1:
        base=D[ms[0]]['keys']
        agree=min(sum(1 for a,b in zip(base,D[m]['keys']) if a==b)/len(base) for m in ms[1:])
    return {'models':ms,'files':{m:D[m]['file'] for m in D},
            'n_raw':{m:D[m]['n'] for m in D},'keys':common,'L':L,
            'unique_keys':ok,'roworder_agreement':agree}
