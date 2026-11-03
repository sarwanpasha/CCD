import json,numpy as np,itertools
from scipy.stats import spearmanr
def lg(*a): print(*a,flush=True)
R16=json.load(open('out/run16.json'))['rows']
A17=json.load(open('out/run17a.json'))['rows']
B17=json.load(open('out/run17b.json'))
R18=json.load(open('out/run18a.json'))['rows']
TASKS=['arc','winogrande','hellaswag','mmlu']
lg('=== A. ACCURACY REGIME (run16, 60 rows) ===')
for tk in TASKS:
  S=[r for r in R16 if r['task']==tk]
  fl=[r['pair'] for r in S if not r['agree_win']]
  cert=[r for r in S if r['nreq_exact'] and r['nreq_exact']<=r['n']]
  lg('%-11s n=%d medrho=%.9f minrho=%.9g maxrho=%.9f medtie=%.4f medcap=%.5f winflip=%d %s certifiable=%d/15'%(
    tk,S[0]['n'],np.median([r['rho_cf'] for r in S]),min(r['rho_cf'] for r in S),max(r['rho_cf'] for r in S),
    np.median([r['p0'] for r in S]),np.median([r['cap_tie'] for r in S]),len(fl),','.join(fl),len(cert)))
lg('winflip total %d/60'%sum(1 for r in R16 if not r['agree_win']))
sg=[r for r in R16 if r['sign_ok'] is not None]
lg('sign test acc regime %d/%d ok; failures=%s'%(sum(1 for r in sg if r['sign_ok']),len(sg),
  [(r['task'],r['pair'],round(r['skew'],4),r['rho_cf']-r['r2']) for r in sg if not r['sign_ok']]))
pos=[r for r in sg if r['skew']>0]
lg('positive-skew rows=%d all_ok=%s'%(len(pos),all(r['sign_ok'] for r in pos)))
d=[r['rho_absdiff'] for r in R16]
lg('closed-form vs numeric max abs diff=%.3e median=%.3e'%(max(d),np.median(d)))
nn=[(r['nreq_chern']/r['nreq_exact']) for r in R16 if r['nreq_exact'] and r['nreq_chern']]
lg('nreq_chernoff/nreq_exact n=%d min=%.4f med=%.4f max=%.4f'%(len(nn),min(nn),np.median(nn),max(nn)))
br=[(r['n']*r['rho_cf'],r['prefactor'],r['br_latt'],r['br_cont']) for r in R16 if r['br_latt'] and r['prefactor']>0]
for lo,hi in [(0,1),(1,5),(5,20),(20,1e9)]:
  s=[x for x in br if lo<=x[0]<hi]
  if s: lg('  nrho in [%g,%g): n=%d median ratio exact/BRlattice=%.4f exact/BRcont=%.4f'%(lo,hi,len(s),
    np.median([x[1]/x[2] for x in s]),np.median([x[1]/x[3] for x in s])))
lg('=== B. DIVERGENCE COMPARISON (run17a, 120 rows) ===')
for reg in ['acc','raw']:
  S=[r for r in A17 if r['reg']==reg and r['rho_kl'] is not None]
  kl=[r['rho_kl'] for r in S]; ch=[r['rho_chi2'] for r in S]; al=[r['alpha_star'] for r in S]
  rt=[r['ratio_kl_chi2'] for r in S]
  eq=sum(1 for r in S if abs(r['rho_chi2_nn']-r['rho_chi2'])<=1e-12*max(1,r['rho_chi2']))
  sgn=sum(1 for r in S if np.sign(r['rho_kl']-r['rho_chi2'])==-np.sign(r['skew']))
  lg('%s n=%d spearman(KL,chi2)=%.6f spearman(KL,alpha)=%.6f ratio KL/chi2 min=%.6f med=%.6f max=%.6f'%(
    reg,len(S),spearmanr(kl,ch).statistic,spearmanr(kl,al).statistic,min(rt),np.median(rt),max(rt)))
  lg('   chi2 nonneg == signed in %d/%d rows; median nn/signed=%.9f max=%.6f'%(eq,len(S),
    np.median([r['nn_over_signed'] for r in S]),max(r['nn_over_signed'] for r in S)))
  lg('   sign(rhoKL-rhochi2)=-sign(skew) in %d/%d rows'%(sgn,len(S)))
  lg('   alpha* pct: min=%.4f med=%.4f max=%.4f | dropfrac med=%.6f | alpha/drop med=%.4f'%(
    100*min(al),100*np.median(al),100*max(al),np.median([r['drop_frac'] for r in S if r['drop_frac']]),
    np.median([r['alpha_star']/r['drop_frac'] for r in S if r['drop_frac']])))
  for tk in TASKS:
    T=[r for r in S if r['task']==tk]
    lg('   %-11s medalpha=%.4f%% medalphak=%d meddrop=%d'%(tk,100*np.median([r['alpha_star'] for r in T]),
      int(np.median([r['alpha_k'] for r in T])),int(np.median([r['drop_k'] for r in T if r['drop_k']]))))
lg('=== C. SUPPORT EXTENSION (run17b, 360 cells) ===')
EPS=B17['eps']
for reg in ['raw','acc']:
  S=[r for r in B17['rows'] if r['reg']==reg]
  dis=[r for r in S if not r['y_agrees']]; agr=[r for r in S if r['y_agrees']]
  lg('%s cells=%d disagreeing=%d agreeing=%d'%(reg,len(S),len(dis),len(agr)))
  for ep in [0.01,0.05,0.10,0.25,0.50]:
    i=EPS.index(ep)
    lg('  eps=%.2f flip_dis=%d/%d flip_agr=%d/%d medrel_dis=%.4f medrel_agr=%.4f minrel=%.4f'%(ep,
      sum(1 for r in dis if r['curve'][i]<=0),len(dis),sum(1 for r in agr if r['curve'][i]<=0),len(agr),
      np.median([r['curve'][i]/r['rho0'] for r in dis if r['rho0']>0]),
      np.median([r['curve'][i]/r['rho0'] for r in agr if r['rho0']>0]),
      min([r['curve'][i]/r['rho0'] for r in S if r['rho0']>0])))
  for tk in TASKS:
    T=[r for r in S if r['novel']==tk]
    i=EPS.index(0.05)
    lg('  novel=%-11s medrel@0.05=%.4f flips@0.05=%d/%d'%(tk,np.median([r['curve'][i]/r['rho0'] for r in T if r['rho0']>0]),
      sum(1 for r in T if r['curve'][i]<=0),len(T)))
lg('=== D. RANK-r (run18a) ===')
for reg in ['raw','acc']:
  S=[r for r in R18 if r['reg']==reg and r.get('ci')]
  lg('%s certificates_with_CI=%d'%(reg,len(S)))
  for r in sorted(S,key=lambda x:(x['task'],x['r'])):
    lg('  %-11s tgt=%-4s r=%d rho=%.8f CI=[%.8f,%.8f] hw/rho=%.4f act=%d S=%s ratio_to_r1=%s'%(
      r['task'],r['target'],r['r'],r['rho'],r['ci'][0],r['ci'][1],r['hw_over_rho'],r['active'],'+'.join(r['subset']),
      ('%.4f'%(r['rho']/[q['rho'] for q in S if q['task']==r['task'] and q['target']==r['target'] and q['r']==1][0]) if any(q['task']==r['task'] and q['target']==r['target'] and q['r']==1 for q in S) else 'NA')))
  L=[r for r in S if r['target']=='MB']
  lg('  MB leader: act==r in %d/%d ; nearest-rival in subset %d/%d ; median hw/rho=%.4f'%(
    sum(1 for r in L if r['active']==r['r']),len(L),sum(1 for r in L if r['minpw_in_subset']),len(L),
    np.median([r['hw_over_rho'] for r in L])))
lg('DONE')
