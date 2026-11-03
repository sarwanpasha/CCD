import sys,json,re,itertools,numpy as np
sys.path.insert(0,'../src')
import ccb,hfio
from loader import MODELS,SHORT,REPO,files_for
from loader2 import rows
L2='meta-llama__Llama-2-7b-hf'
OTH=[m for m in MODELS if m!=L2]
A_DATE='2023-08-19T16:35:46.942696'
B_DATE='2023-08-29'
rng=np.random.default_rng(7)

def pick(model,pat,date=None,exclude=None):
    c=files_for(model,pat)
    if date: c=[x for x in c if date in x[0]]
    if exclude: c=[x for x in c if exclude not in x[0]]
    best=None
    for p,s in c:
        try: t=hfio.table(REPO+model,p)
        except Exception: continue
        if t.num_rows<50: continue
        if best is None or t.num_rows>best[1]: best=(p,t.num_rows,t)
    return best

def l2_dates(pat):
    ds=set()
    for p,s in files_for(L2,pat):
        mm=re.search(r'(\d{4}-\d{2}-\d{2}T[\d:.\-]+)',p)
        if mm: ds.add(mm.group(1))
    return sorted(ds)

tasks=[('arc',r'arc.challenge\|25'),('hellaswag',r'hellaswag\|10')]
tree=hfio.tree(REPO+L2)
sub=sorted({re.search(r'hendrycksTest-([A-Za-z_\-]+)\|5',p).group(1) for p,s in tree if 'hendrycksTest-' in p and '|5' in p})
tasks+= [('mmlu:'+s,r'hendrycksTest-'+re.escape(s)+r'\|5') for s in sub]
print('ntasks',len(tasks),flush=True)

OBS=[];SKIP=[]
for tname,pat in tasks:
    ds=l2_dates(pat)
    dA=[d for d in ds if d.startswith(A_DATE[:10])]
    dB=[d for d in ds if d.startswith(B_DATE)]
    if not dA or not dB: SKIP.append((tname,'dates',ds)); continue
    dA=sorted(dA)[0]; dB=sorted(dB)[-1]
    a=pick(L2,pat,dA); b=pick(L2,pat,dB)
    if a is None or b is None or a[1]!=b[1]: SKIP.append((tname,'rows',None)); continue
    kA,vA=rows(a[2],'raw'); kB,vB=rows(b[2],'raw')
    if kA!=kB: SKIP.append((tname,'keys',None)); continue
    O={}
    ok=True
    for m in OTH:
        r=pick(m,pat)
        if r is None or r[1]!=a[1]: ok=False; break
        k,v=rows(r[2],'raw')
        if k!=kA: ok=False; break
        O[m]=v
    if not ok: SKIP.append((tname,'others',None)); continue
    for m in OTH:
        for tag,l2v in [('A',vA)]:
            if l2v.mean()<=O[m].mean(): W,Lo,gA=('L2',SHORT[m]),None,l2v-O[m]
            else: gA=O[m]-l2v
            win_is_l2 = l2v.mean()<=O[m].mean()
            g=gA
            c=ccb.ccb(g)
            if not np.isfinite(c.get('rho',np.nan)): SKIP.append((tname+'/'+SHORT[m],'deg',None)); continue
            gB = (vB-O[m]) if win_is_l2 else (O[m]-vB)
            rev = bool(gB.mean()>=0)
            OBS.append({'task':tname,'pair':'L2:'+SHORT[m],'cluster':tname,'n':int(g.size),
                        'l2_is_winner':bool(win_is_l2),'m':-c['mean'],'rho':c['rho'],
                        'half_t2':c['half_t2'],'skew':c['skew'],'beta':c['beta'],'tau':c['tau'],
                        'margin_drift':float(gB.mean()-g.mean()),'reversal':rev,
                        'dateA':dA,'dateB':dB})
    print('task',tname,'obs',len(OBS),flush=True)

def auc(score,lab):
    score=np.asarray(score,float); lab=np.asarray(lab,bool)
    p=score[lab]; q=score[~lab]
    if p.size==0 or q.size==0: return float('nan')
    allv=np.concatenate([p,q]); r=allv.argsort().argsort()+1.0
    # average ranks for ties
    import scipy.stats as st
    r=st.rankdata(allv)
    return float((r[:p.size].sum()-p.size*(p.size+1)/2)/(p.size*q.size))

lab=[o['reversal'] for o in OBS]
sc={'rho':[o['rho'] for o in OBS],'half_t2':[o['half_t2'] for o in OBS],'m':[o['m'] for o in OBS]}
AUC={k:auc([-v for v in s],lab) for k,s in sc.items()}
clus=sorted({o['cluster'] for o in OBS}); ci={}
B=4000
boot={k:[] for k in sc}
boot['rho_minus_half']=[];boot['rho_minus_m']=[]
idx={c:[i for i,o in enumerate(OBS) if o['cluster']==c] for c in clus}
for b in range(B):
    pickc=rng.choice(len(clus),len(clus),replace=True)
    ii=[i for c in pickc for i in idx[clus[c]]]
    lb=[lab[i] for i in ii]
    if not any(lb) or all(lb): continue
    a1=auc([-sc['rho'][i] for i in ii],lb); a2=auc([-sc['half_t2'][i] for i in ii],lb); a3=auc([-sc['m'][i] for i in ii],lb)
    boot['rho'].append(a1);boot['half_t2'].append(a2);boot['m'].append(a3)
    boot['rho_minus_half'].append(a1-a2);boot['rho_minus_m'].append(a1-a3)
for k,v in boot.items():
    v=np.array(v)
    if v.size<10: ci[k]=None; continue
    ci[k]=[float(np.percentile(v,2.5)),float(np.percentile(v,97.5)),float(v.mean()),int(v.size)]
# permutation null: shuffle reversal labels within clusters
null=[]
for b in range(2000):
    lb=list(lab)
    for c in clus:
        ii=idx[c]; vals=[lb[i] for i in ii]; rng.shuffle(vals)
        for i,v in zip(ii,vals): lb[i]=v
    if any(lb) and not all(lb): null.append(auc([-x for x in sc['rho']],lb))
null=np.array(null) if null else np.array([float('nan')])
out={'n_obs':len(OBS),'n_clusters':len(clus),'n_reversals':int(sum(lab)),
     'clusters_with_reversal':len({o['cluster'] for o in OBS if o['reversal']}),
     'AUC':AUC,'boot_ci':ci,'null_auc_mean':float(null.mean()),
     'null_auc_ci':[float(np.percentile(null,2.5)),float(np.percentile(null,97.5))],
     'skip':SKIP[:40],'obs':OBS}
with open('out/run3.json','w') as f: json.dump(out,f,indent=1,default=float)
print('WROTE run3.json',len(OBS),AUC)
