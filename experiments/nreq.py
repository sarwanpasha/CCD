import json,math,statistics as st
d=json.load(open('out/run4.json'))['pairs']
L=math.log(20.0)
print('certified exponent n*rho, Chernoff bound exp(-n rho), and items needed for delta=0.05')
for mo in ['raw','softmax','lennorm']:
    P_=[p for p in d if p['mode']==mo and p['rho']]
    print('=== %s ==='%mo)
    nc=0
    for p in sorted(P_,key=lambda x:x['rho']):
        e=p['n']*p['rho']; bd=math.exp(-e); need=L/p['rho']; extra=max(0.0,need-p['n'])
        cert='YES' if e>=L else 'no '
        if e>=L: nc+=1
        print('  %-10s %-4s>%-4s n=%5d rho=%.6f  n*rho=%10.2f  exp(-n rho)=%.3e  cert@.05=%s  n_needed=%12.0f  extra=%12.0f'%(p['task'],p['winner'],p['loser'],p['n'],p['rho'],e,bd,cert,need,extra))
    print('  certified at delta=0.05: %d / %d'%(nc,len(P_)))
    ex=[max(0.0,L/p['rho']-p['n']) for p in P_]
    print('  extra items needed: min %.0f  median %.0f  max %.0f'%(min(ex),st.median(ex),max(ex)))
