exec(open('run9b.py').read().split("_TC={}")[0])
import numpy as np, json, re, time, itertools, hashlib
_TC={}; _o=hfio.tree
def _tree(r):
    if r not in _TC: _TC[r]=_o(r)
    return _TC[r]
hfio.tree=_tree
t0=time.time()
paths=[p for p,s in _tree(REPO+MODELS[0])]
SUB=sorted(set(re.search(r'hendrycksTest-(.+?)\|5',p).group(1) for p in paths
    if 'hendrycksTest-' in p and p.endswith('.parquet')))
print('SUBJECTS_IN_TREE %d'%len(SUB),flush=True)
def acct(subj):
    per={}; nrow={}
    for m in MODELS:
        c=files_for(m,r'hendrycksTest-'+re.escape(subj)+r'\|5')
        if not c: return None
        best=None
        for p,sz in c:
            try: t=hfio.table(REPO+m,p)
            except Exception: continue
            if best is None or t.num_rows>best.num_rows: best=t
        if best is None: return None
        d=best.to_pydict()
        nrow[m]=len(d['example']); per[m]=set(ekey(e) for e in d['example'])
    it=set.intersection(*[per[m] for m in MODELS]); un=set.union(*[per[m] for m in MODELS])
    return nrow,{m:len(per[m]) for m in MODELS},len(it),len(un)
TOTn={m:0 for m in MODELS}; TOTd={m:0 for m in MODELS}; TI=0; TU=0; rows=[]
for s in SUB:
    try: r=acct(s)
    except Exception as e:
        print('ERR',s,repr(e)[:60],flush=True); continue
    if r is None: continue
    nrow,ndis,ni,nu=r
    for m in MODELS: TOTn[m]+=nrow[m]; TOTd[m]+=ndis[m]
    TI+=ni; TU+=nu
    rows.append({'s':s,'nrow':{SHORT[m]:nrow[m] for m in MODELS},'ndis':{SHORT[m]:ndis[m] for m in MODELS},'inter':ni,'union':nu})
print('=== MMLU INTERSECTION ACCOUNTING (%d subjects, %.1fs)'%(len(rows),time.time()-t0),flush=True)
for m in MODELS: print('  %-6s raw_rows=%d distinct_hashes=%d'%(SHORT[m],TOTn[m],TOTd[m]),flush=True)
print('  SUM_OVER_SUBJECTS intersection=%d union=%d'%(TI,TU),flush=True)
print('  dropped_vs_union=%d  subjects_with_any_mismatch=%d'%(TU-TI,sum(1 for r in rows if r['inter']!=r['union'])),flush=True)
for r in rows:
    if r['inter']!=r['union']: print('   MISMATCH %-34s inter=%d union=%d rows=%s'%(r['s'],r['inter'],r['union'],r['nrow']),flush=True)
json.dump(rows,open('out/run14_acct.json','w'))
D=json.load(open('out/run9.json')); R=D['recs']; LOG=np.log(1.0/0.05)
print('=== MMLU CERTIFY-HALF RADII, per pair, delta=0.05 thresh=log(1/d)=%.6f'%LOG,flush=True)
PAIRS=sorted(set(r['p'] for r in R))
for arm in ('domain','iid'):
    print('-- arm=%s'%arm,flush=True)
    print('%-10s %5s %9s %9s %9s %9s %6s %9s %9s %7s'%('pair','nsp','rho_min','rho_q25','rho_med','rho_max','cert','rmin_cert','rmed_cert','rev|c'),flush=True)
    allc=0; alln=0; allrev=0
    for p in PAIRS:
        S=[r for r in R if r['p']==p and r['arm']==arm]
        rv=np.array([r['rho'] if r['rho'] is not None else np.inf for r in S],float)
        nB=np.array([r['nB'] for r in S],float)
        rev=np.array([r['rev'] for r in S],int)
        c=(nB*rv>=LOG)
        fin=np.isfinite(rv)
        rc=rv[c&fin]
        allc+=int(c.sum()); alln+=len(S); allrev+=int(rev[c].sum())
        print('%-10s %5d %9.6f %9.6f %9.6f %9.6f %4d/%d %9.6f %9.6f %7d'%(p,len(S),
            np.nanmin(rv[fin]) if fin.any() else -1,np.percentile(rv[fin],25) if fin.any() else -1,
            np.median(rv[fin]) if fin.any() else -1,np.nanmax(rv[fin]) if fin.any() else -1,
            int(c.sum()),len(S),rc.min() if len(rc) else -1,np.median(rc) if len(rc) else -1,int(rev[c].sum())),flush=True)
    print('  TOTAL arm=%s certified=%d/%d rev_among_certified=%d rate=%.5f'%(arm,allc,alln,allrev,allrev/max(1,allc)),flush=True)
    ALL=[r for r in R if r['arm']==arm]
    rv=np.array([r['rho'] if r['rho'] is not None else np.inf for r in ALL],float)
    nB=np.array([r['nB'] for r in ALL],float); rev=np.array([r['rev'] for r in ALL],int)
    c=nB*rv>=LOG
    print('  pooled median certify-half rho over all splits = %.6f ; median among certified = %.6f ; nB median=%d'%(
        np.median(rv[np.isfinite(rv)]),np.median(rv[c&np.isfinite(rv)]),int(np.median(nB))),flush=True)
    print('  smallest certified rho = %.8f  (needs nB>=%.0f)'%(rv[c&np.isfinite(rv)].min(),LOG/rv[c&np.isfinite(rv)].min()),flush=True)
    print('  rev|uncertified = %.5f (%d/%d)'%(rev[~c].mean(),int(rev[~c].sum()),int((~c).sum())),flush=True)
print('=== MMLU POOLED per-pair rho, full precision (from run13)',flush=True)
T=json.load(open('out/run13.json'))
for r in T:
    if r['task']!='mmlu': continue
    if r.get('rho') is None: print('%-10s INF'%r['pair'],flush=True); continue
    ci=r['ci']; hw=0.5*(ci[1]-ci[0])
    print('%-10s n=%d win=%-6s rho=%.10f  CI=[%.10f, %.10f] hw=%.10f hw/rho=%.4f sd_b=%.3e s=%.5f t=%.4f skew=%+.4f'%(
        r['pair'],r['n'],r['winner'],r['rho'],ci[0],ci[1],hw,hw/r['rho'],r['boot_sd'],r['s'],r['t'],r['skew']),flush=True)
print('=== target pair detail',flush=True)
for r in T:
    if r['task']=='mmlu' and r['pair'] in ('L2:MI','MI:L2'):
        print(json.dumps(r,indent=1),flush=True)
print('DONE %.1fs'%(time.time()-t0),flush=True)
