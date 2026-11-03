import sys,json,numpy as np
sys.path.insert(0,'../src')
import hfio,loader
from loader import MODELS,SHORT,best_file
O={}
for task,pat,want in [('arc',r'arc.challenge',1172),('hellaswag',r'hellaswag',10042)]:
 O[task]={}
 for m in MODELS:
  p,t=best_file(m,pat,want); d=t.to_pydict()
  ch=d['choices']; go=d['gold']; ct=d['cont_tokens']
  n=len(ch); eq=0;e1=0;em=0; ex=[]
  for i in range(n):
   gi=go[i]
   if isinstance(gi,(list,tuple)): gi=gi[0]
   gi=int(gi); L=len(ct[i][gi]); s=str(ch[i][gi]); C=len(s)
   if L==C: eq+=1
   if L==C+1: e1+=1
   if L==C-1: em+=1
   if i<2: ex.append([L,C,s[:60],str(ct[i][gi][:6])[:90]])
  O[task][SHORT[m]]={'n':n,'eq':eq,'plus1':e1,'minus1':em,'ex':ex}
  print(task,SHORT[m],'len==chars %d/%d  +1 %d  -1 %d'%(eq,n,e1,em),flush=True)
json.dump(O,open('out/run7.json','w'),indent=1,default=str)
print('WROTE')
