import sys,os,json,time,re,hashlib,itertools
sys.path.insert(0,'../src')
import numpy as np, hfio
from loader import MODELS,SHORT,REPO,files_for,best_file
from loader2 import ckey
from scipy.stats import skew as SKW, kurtosis as KRT, binom, spearmanr
from scipy.special import logsumexp
_TC={}; _o=hfio.tree
def _tree(r):
  if r not in _TC: _TC[r]=_o(r)
  return _TC[r]
hfio.tree=_tree
def ekey(s): return hashlib.md5(str(s).encode()).hexdigest()
def gi_of(x):
  if isinstance(x,(list,tuple)): x=x[0]
  return int(x)
T0=time.time()
def lg(*a): print(*a,flush=True)
def load_ch(task,pat,want):
  R={}
  for m in MODELS:
    p,t=best_file(m,pat,want); d=t.to_pydict()
    ch=d['choices']; gd=d['gold']; pr=d['predictions']
    K={}
    for i in range(len(ch)):
      g=gi_of(gd[i]); v=np.asarray(pr[i],dtype=np.float64)
      K[ckey(ch[i],g)]=(float(-v[g]),1.0 if int(np.argmax(v))==g else 0.0)
    R[m]=K; lg('  ld',task,SHORT[m],len(K),'%.0fs'%(time.time()-T0))
  com=sorted(set.intersection(*[set(R[m]) for m in MODELS]))
  L=np.array([[R[m][k][0] for k in com] for m in MODELS])
  C=np.array([[R[m][k][1] for k in com] for m in MODELS])
  return L,C,np.zeros(len(com),dtype=np.int32)
def load_wino():
  R={}
  for m in MODELS:
    d=None
    for p,sz in files_for(m,r'winogrande\|5'):
      dd=hfio.table(REPO+m,p).to_pydict()
      if set(len(x) for x in dd['predictions'])=={2}:
        if d is None or len(dd['example'])>len(d['example']): d=dd
    ex=d['example']; o1=d['option1']; o2=d['option2']; an=d['answer']; pr=d['predictions']
    K={}
    for i in range(len(ex)):
      g=int(str(an[i]).strip())-1; v=np.asarray(pr[i],dtype=np.float64)
      K[ekey(str(ex[i])+'##'+str(o1[i])+'##'+str(o2[i]))]=(float(-v[g]),1.0 if int(np.argmax(v))==g else 0.0)
    R[m]=K; lg('  ld wino',SHORT[m],len(K),'%.0fs'%(time.time()-T0))
  com=sorted(set.intersection(*[set(R[m]) for m in MODELS]))
  L=np.array([[R[m][k][0] for k in com] for m in MODELS])
  C=np.array([[R[m][k][1] for k in com] for m in MODELS])
  return L,C,np.zeros(len(com),dtype=np.int32)
def load_mmlu():
  tr=hfio.tree(REPO+MODELS[0])
  subs=sorted(set(re.findall(r'hendrycksTest-([a-z_]+)\|5',' '.join(p for p,s in tr))))
  lg('  mmlu subjects',len(subs))
  Ls=[]; Cs=[]; cl=[]; SB=[]
  for si,sb in enumerate(subs):
    R={}; ok=True
    for m in MODELS:
      c=files_for(m,r'hendrycksTest-'+re.escape(sb)+r'\|5')
      if not c: ok=False; break
      best=None
      for p,sz in c:
        try: tt=hfio.table(REPO+m,p)
        except Exception: continue
        if best is None or tt.num_rows>best.num_rows: best=tt
      if best is None: ok=False; break
      d=best.to_pydict(); ex=d['example']; gd=d['gold']; pr=d['predictions']
      K={}
      for i in range(len(ex)):
        g=gi_of(gd[i]); v=np.asarray(pr[i],dtype=np.float64)
        K[ekey(ex[i])]=(float(-v[g]),1.0 if int(np.argmax(v))==g else 0.0)
      R[m]=K
    if not ok: lg('  SKIP',sb); continue
    com=sorted(set.intersection(*[set(R[m]) for m in MODELS]))
    Ls.append(np.array([[R[m][k][0] for k in com] for m in MODELS]))
    Cs.append(np.array([[R[m][k][1] for k in com] for m in MODELS]))
    cl.append(np.full(len(com),len(SB),dtype=np.int32)); SB.append(sb)
    if si%12==0: lg('  mmlu',si,sb,len(com),'%.0fs'%(time.time()-T0))
  lg('  mmlu subjects used',len(SB))
  return np.concatenate(Ls,axis=1),np.concatenate(Cs,axis=1),np.concatenate(cl)
exec(open('run10.py').read().split('def fsbound')[0])
CF='cache_lc.npz'
if os.path.exists(CF):
  z=np.load(CF); DAT={k:z[k] for k in z.files}; lg('cache hit',CF)
else:
  DAT={}
  for nm,pat,want in [('arc',r'arc.challenge',1172),('hellaswag',r'hellaswag',10042)]:
    L,C,cl=load_ch(nm,pat,want); DAT['L_'+nm]=L; DAT['C_'+nm]=C; DAT['cl_'+nm]=cl; lg('TASK',nm,L.shape)
  L,C,cl=load_wino(); DAT['L_winogrande']=L; DAT['C_winogrande']=C; DAT['cl_winogrande']=cl; lg('TASK wino',L.shape)
  L,C,cl=load_mmlu(); DAT['L_mmlu']=L; DAT['C_mmlu']=C; DAT['cl_mmlu']=cl; lg('TASK mmlu',L.shape)
  np.savez_compressed(CF,**DAT); lg('WROTE',CF,'%.0fs'%(time.time()-T0))
TASKS=['arc','winogrande','hellaswag','mmlu']
def rho_cf(pp,pm,p0):
  if pp>=pm: return 0.0
  Z=p0+2.0*float(np.sqrt(pp*pm))
  return float(-np.log(Z)) if Z>0 else float('inf')
def erev(n,pp,pm):
  pd=pp+pm
  if pd<=0: return 0.0
  q=pp/pd; m=np.arange(0,n+1,dtype=np.float64)
  lp=binom.logpmf(m,n,pd); c=np.ceil(m/2.0)
  sf=binom.sf(c-1.0,m,q); sf=np.where(m==0,1.0,sf)
  ok=sf>0
  if not ok.any(): return -np.inf
  return float(logsumexp(lp[ok]+np.log(sf[ok])))
def nreq_ex(pp,pm,delta=0.05):
  tgt=float(np.log(delta)); hi=1
  while erev(hi,pp,pm)>tgt:
    hi*=2
    if hi>2097152: return None
  lo=1
  while lo<hi:
    md=(lo+hi)//2
    if erev(md,pp,pm)<=tgt: hi=md
    else: lo=md+1
  return int(lo)
def boot_cf(h,cl,B=4000,seed=11):
  rng=np.random.default_rng(seed); n=len(h); out=np.empty(B); uc=np.unique(cl)
  if len(uc)>1:
    idxs=[np.where(cl==u)[0] for u in uc]; K=len(idxs)
    for b in range(B):
      x=h[np.concatenate([idxs[j] for j in rng.integers(0,K,K)])]
      pp=float((x>0).mean()); pm=float((x<0).mean()); out[b]=rho_cf(pp,pm,1.0-pp-pm)
  else:
    for b in range(B):
      x=h[rng.integers(0,n,n)]
      pp=float((x>0).mean()); pm=float((x<0).mean()); out[b]=rho_cf(pp,pm,1.0-pp-pm)
  return out
ROWS=[]
for tk in TASKS:
  C=DAT['C_'+tk]; L=DAT['L_'+tk]; cl=DAT['cl_'+tk]; n=C.shape[1]
  lg('== task',tk,'n=',n,'acc=',['%s %.4f'%(SHORT[MODELS[i]],C[i].mean()) for i in range(6)])
  for ia,ib in itertools.combinations(range(6),2):
    nm=SHORT[MODELS[ia]]+':'+SHORT[MODELS[ib]]
    g=(C[ib]-C[ia]).astype(np.float64)
    sg=-1.0 if g.mean()>0 else 1.0; h=sg*g
    win=SHORT[MODELS[ia]] if sg>0 else SHORT[MODELS[ib]]
    pp=float((h>0).mean()); pm=float((h<0).mean()); p0=1.0-pp-pm
    rcf=rho_cf(pp,pm,p0); rnum,bnum=rho(h)
    bcf=float(0.5*np.log(pm/pp)) if pp>0 else float('inf')
    Zc=p0+2.0*float(np.sqrt(pp*pm)); sb2=2.0*float(np.sqrt(pp*pm))/Zc if Zc>0 else 0.0
    mu=float(h.mean()); sd=float(h.std(ddof=0)); sk=float(SKW(h)); ku=float(KRT(h))
    r2=mu*mu/(2.0*sd*sd); tst=float(-mu/(sd/np.sqrt(n)))
    lex=erev(n,pp,pm); lch=-n*rcf
    brc=float(1.0/(bnum*np.sqrt(sb2)*np.sqrt(2*np.pi*n))) if (pp>0 and sb2>0 and bnum>0) else None
    brl=float(1.0/((1.0-np.exp(-bnum))*np.sqrt(sb2)*np.sqrt(2*np.pi*n))) if (pp>0 and sb2>0 and bnum>0) else None
    v=boot_cf(h,cl); lo=float(np.percentile(v,2.5)); hi=float(np.percentile(v,97.5))
    kp=int((h>0).sum()); km=int((h<0).sum())
    mcn=float(min(1.0,2.0*binom.cdf(min(kp,km),kp+km,0.5))) if kp+km>0 else 1.0
    ne=nreq_ex(pp,pm) if rcf>0 else None
    nc=float(np.log(20.0)/rcf) if rcf>0 else None
    gr=L[ia]-L[ib]; sgr=-1.0 if gr.mean()>0 else 1.0; hr=sgr*gr
    winr=SHORT[MODELS[ia]] if sgr>0 else SHORT[MODELS[ib]]
    rr,brr=rho(hr)
    rec={'task':tk,'pair':nm,'n':n,'win_acc':win,'win_raw':winr,'agree_win':win==winr,
      'acc_a':float(C[ia].mean()),'acc_b':float(C[ib].mean()),
      'pp':pp,'pm':pm,'p0':p0,'kp':kp,'km':km,'rho_cf':rcf,'rho_num':float(rnum),
      'rho_absdiff':abs(float(rnum)-rcf),'cap_tie':float(-np.log(p0)) if p0>0 else float('inf'),
      'beta_cf':bcf,'beta_num':float(bnum),'sig2_tilt':sb2,'mu':mu,'sd':sd,'skew':sk,'exkurt':ku,
      'r2':r2,'t':tst,'ci':[lo,hi],'boot_sd':float(v.std(ddof=1)),'hw_over_rho':float((hi-lo)/2.0/rcf) if rcf>0 else None,
      'log_pexact':lex,'log_chernoff':lch,'prefactor':float(np.exp(lex-lch)),
      'br_cont':brc,'br_latt':brl,'nreq_exact':ne,'nreq_chern':nc,
      'mcnemar_p':mcn,'rho_raw':float(rr) if np.isfinite(rr) else None,'sign_ok':bool(np.sign(rcf-r2)==-np.sign(sk)) if sk!=0 else None}
    ROWS.append(rec)
    lg('%-11s %-8s pp=%.6f pm=%.6f p0=%.6f rho=%.9f dcf=%.2e cap=%.5f t=%7.2f r2=%.9f sk=%+.4f pref=%.4f brl=%s CI=[%.6f,%.6f] ne=%s nc=%s rraw=%.7f %s'%(
      nm,tk,pp,pm,p0,rcf,abs(float(rnum)-rcf),(-np.log(p0) if p0>0 else float('inf')),tst,r2,sk,float(np.exp(lex-lch)),
      ('%.4f'%brl if brl else 'NA'),lo,hi,ne,('%.0f'%nc if nc else 'NA'),float(rr),'' if win==winr else 'WINFLIP(raw=%s)'%winr))
json.dump({'rows':ROWS},open('out/run16.json','w'))
lg('WROTE out/run16.json rows=%d %.0fs'%(len(ROWS),time.time()-T0))
