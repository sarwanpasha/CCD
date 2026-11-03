import json, numpy as np
A=json.load(open('out/run10.json'))['cases']
B=json.load(open('out/run13.json'))
ROWS=[]
for r in A:
    if r['conv']!='raw' or r.get('rho') is None: continue
    ROWS.append(('arc' if r['task']=='arc' else 'hellaswag',r))
for r in B:
    if r.get('rho') is None: continue
    ROWS.append((r['task'],r))
print('=== FULL-PRECISION SIGN AUDIT, 4 benchmarks, raw convention, %d rows'%len(ROWS))
print('%-11s %-8s %6s %-18s %-18s %-13s %-11s %-9s %s'%('bench','pair','n','rho_star','half_t2=r2','rho-r2','skew','rel_diff','ok'))
ok=0; fails=[]
for tk,r in ROWS:
    rho=r['rho']; r2=r['r2']; sk=r['skew']; d=rho-r2
    good=(np.sign(d)==-np.sign(sk)) or d==0
    ok+=int(good)
    if not good: fails.append((tk,r['pair'],rho,r2,d,sk))
    print('%-11s %-8s %6d %.12f %.12f %+.6e %+.6f %+.3e %s'%(tk,r['pair'],r['n'],rho,r2,d,sk,d/rho,'Y' if good else 'N'))
print('SIGN TEST %d/%d'%(ok,len(ROWS)))
print('FAILURES:')
for f in fails: print('  %-11s %-8s rho=%.12f r2=%.12f diff=%+.6e skew=%+.6f rel=%+.3e'%(f[0],f[1],f[2],f[3],f[4],f[5],f[4]/f[2]))
neg=[(tk,r) for tk,r in ROWS if r['skew']<0]; pos=[(tk,r) for tk,r in ROWS if r['skew']>=0]
def cnt(L): return sum(1 for tk,r in L if np.sign(r['rho']-r['r2'])==-np.sign(r['skew'])),len(L)
print('negative-skew %d/%d ; positive-skew %d/%d'%(cnt(neg)+cnt(pos)))
for lo,hi in ((0,0.1),(0.1,0.3),(0.3,0.6),(0.6,2.0)):
    S=[(tk,r) for tk,r in ROWS if lo<=abs(r['s'])<hi]
    if S: print('  s in [%.1f,%.1f): %d/%d'%((lo,hi)+cnt(S)))
for lo,hi in ((0,0.05),(0.05,0.2),(0.2,0.5),(0.5,2.0)):
    S=[(tk,r) for tk,r in ROWS if lo<=abs(r['skew'])<hi]
    if S: print('  |skew| in [%.2f,%.2f): %d/%d'%((lo,hi)+cnt(S)))
print('=== ADJUDICATION of the three queried rows')
for tk,pr in (('winogrande','FA:L2'),('mmlu','L2:MB'),('mmlu','L2:MI'),('mmlu','MI:L2'),('winogrande','L2:FA')):
    for t2,r in ROWS:
        if t2==tk and r['pair']==pr:
            print('%-11s %-8s n=%d'%(tk,pr,r['n']))
            print('   rho_star   = %.15g'%r['rho'])
            print('   half_t2    = %.15g'%r['r2'])
            print('   rho - r2   = %.15g'%(r['rho']-r['r2']))
            print('   s          = %.15g'%r['s'])
            print('   t          = %.15g'%r['t'])
            print('   mu         = %.15g'%r['mu'])
            print('   sd         = %.15g'%r['sd'])
            print('   skew       = %.15g'%r['skew'])
            print('   exkurt     = %.15g'%r['exkurt'])
            print('   pred_o3    = %.15g   (r2 - skew*s^3/6 sign convention as reported: r3=%s)'%(r['r2']-r['skew']*abs(r['s'])**3/6.0,r.get('r3')))
            print('   CI         = [%.15g, %.15g]'%(r['ci'][0],r['ci'][1]))
            print('   boot_sd    = %.15g'%r['boot_sd'])
            print('   B          = %.15g'%r.get('B',r.get('Bmax',float('nan'))))
