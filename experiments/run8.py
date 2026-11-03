import sys,json,collections
sys.path.insert(0,'../src')
import loader
from loader import MODELS,SHORT,best_file
O={}
for task,pat,want in [('arc',r'arc.challenge',1172),('hellaswag',r'hellaswag',10042)]:
 O[task]={}
 for m in MODELS:
  p,t=best_file(m,pat,want); d=t.to_pydict()
  ch=d['choices']; go=d['gold']; ct=d['cont_tokens']
  gi=go[0]
  if isinstance(gi,(list,tuple)): gi=gi[0]
  gi=int(gi)
  full=list(ct[0][gi])
  tail=collections.Counter(full[6:]).most_common(4)
  nsub=len(ct[0]); lens=[len(x) for x in ct[0]]
  allsame=len(set(lens))==1
  O[task][SHORT[m]]={'n_sublists':nsub,'n_choices':len(ch[0]),'sublist_lens':lens,
    'all_equal':allsame,'gold_first12':full[:12],'gold_last8':full[-8:],
    'tail_counter':tail,'total_len':len(full)}
  print(task,SHORT[m],'sublists=%d choices=%d lens=%s'%(nsub,len(ch[0]),lens),flush=True)
json.dump(O,open('out/run8.json','w'),indent=1,default=str)
print('WROTE')
