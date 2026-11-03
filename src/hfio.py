import urllib.parse
import json,os,urllib.request,io
import pyarrow.parquet as pq
API='https://huggingface.co/api/datasets/'
RESOLVE='https://huggingface.co/datasets/{repo}/resolve/main/{path}'
CACHE=os.path.join(os.environ.get('CCD_ROOT', '.'), 'data')

def _get(url):
    req=urllib.request.Request(url,headers={'User-Agent':'ccb/1.0'})
    with urllib.request.urlopen(req,timeout=120) as r:
        return r.read()

def tree(repo):
    out=[];cur=None
    url=API+repo+'/tree/main?recursive=true'
    data=json.loads(_get(url).decode())
    for e in data:
        if e.get('type')=='file':
            out.append((e['path'],e.get('size',0)))
    return out

def fetch(repo,path):
    safe=(repo+'/'+path).replace('/','__').replace('|','_').replace(':','_')
    fp=os.path.join(CACHE,safe)
    if os.path.exists(fp) and os.path.getsize(fp)>0:
        return fp
    url=RESOLVE.format(repo=repo,path=urllib.parse.quote(path))
    b=_get(url)
    with open(fp,'wb') as f: f.write(b)
    return fp

def table(repo,path):
    return pq.read_table(fetch(repo,path))
