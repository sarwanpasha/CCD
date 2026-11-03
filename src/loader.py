import os
import re,numpy as np,hfio
MODELS=['tiiuae__falcon-7b','meta-llama__Llama-2-7b-hf','lmsys__vicuna-7b-v1.5',
        'mistralai__Mistral-7B-Instruct-v0.1','mistralai__Mistral-7B-v0.1','lmsys__vicuna-7b-v1.3']
SHORT={'tiiuae__falcon-7b':'FA','meta-llama__Llama-2-7b-hf':'L2',
       'lmsys__vicuna-7b-v1.5':'V15','mistralai__Mistral-7B-Instruct-v0.1':'MI',
       'mistralai__Mistral-7B-v0.1':'MB','lmsys__vicuna-7b-v1.3':'V13'}
REPO='open-llm-leaderboard-old/details_'

def files_for(model,taskpat):
    t=hfio.tree(REPO+model)
    return [(p,s) for p,s in t if re.search(taskpat,p) and p.endswith('.parquet')]

def best_file(model,taskpat,want_rows=None):
    cand=files_for(model,taskpat)
    if not cand: return None,None
    best=None
    for p,s in cand:
        try: t=hfio.table(REPO+model,p)
        except Exception: continue
        nr=t.num_rows
        if want_rows is not None and nr!=want_rows: continue
        if best is None or nr>best[1]: best=(p,nr,t)
    if best is None:
        for p,s in cand:
            t=hfio.table(REPO+model,p)
            if best is None or t.num_rows>best[1]: best=(p,t.num_rows,t)
    return best[0],best[2]

def losses(tab,mode='raw'):
    d=tab.to_pydict()
    ex=d['example']; gold=d['gold']; pred=d['predictions']; ct=d.get('cont_tokens')
    out=np.empty(len(ex)); keys=[]
    for i in range(len(ex)):
        pr=np.asarray(pred[i],dtype=np.float64)
        gi=gold[i]
        if isinstance(gi,(list,tuple)): gi=gi[0]
        gi=int(gi)
        if mode=='raw':
            v=-pr[gi]
        elif mode=='softmax':
            mx=pr.max(); lse=mx+np.log(np.exp(pr-mx).sum()); v=-(pr[gi]-lse)
        elif mode=='lennorm':
            L=np.array([max(len(c),1) for c in ct[i]],dtype=np.float64)
            pn=pr/L; v=-pn[gi]
        else: raise ValueError(mode)
        out[i]=v; keys.append(str(ex[i]))
    return keys,out

def load_task(taskpat,mode='raw',want_rows=None,models=None):
    models=models or MODELS
    D={}
    for m in models:
        p,t=best_file(m,taskpat,want_rows)
        if t is None: continue
        k,v=losses(t,mode)
        D[m]={'file':p,'keys':k,'loss':v,'n':len(k)}
    common=None
    for m in D:
        s=set(D[m]['keys'])
        common=s if common is None else (common & s)
    common=sorted(common)
    idx={m:{k:i for i,k in enumerate(D[m]['keys'])} for m in D}
    L={m:np.array([D[m]['loss'][idx[m][k]] for k in common]) for m in D}
    return {'models':list(D.keys()),'files':{m:D[m]['file'] for m in D},
            'n_raw':{m:D[m]['n'] for m in D},'keys':common,'L':L}
