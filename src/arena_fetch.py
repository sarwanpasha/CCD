"""Download LMArena human-preference battles, keep only the ranking-relevant
columns, and drop the raw shards. Also grab the published leaderboard snapshot."""
import json, os, urllib.request, urllib.parse, time
import pyarrow.parquet as pq
import pyarrow as pa

ROOT = os.environ.get('CCD_ROOT', '.')
OUT  = os.path.join(ROOT, 'data_arena')
os.makedirs(OUT, exist_ok=True)
API  = 'https://huggingface.co/api/datasets/'
RES  = 'https://huggingface.co/datasets/{r}/resolve/main/{p}'
KEEP = ['id', 'model_a', 'model_b', 'winner', 'language', 'is_code', 'timestamp']

def get(u, tries=4):
    last = None
    for k in range(tries):
        try:
            req = urllib.request.Request(u, headers={'User-Agent': 'ccb/1.0'})
            with urllib.request.urlopen(req, timeout=600) as r:
                return r.read()
        except Exception as e:
            last = e; time.sleep(3 + 5*k)
    raise last

def tree(repo):
    t = json.loads(get(API + repo + '/tree/main?recursive=true').decode())
    return [(e['path'], e.get('size', 0)) for e in t if e.get('type') == 'file']

def battles():
    repo = 'lmarena-ai/arena-human-preference-140k'
    parts = []
    for path, size in sorted(tree(repo)):
        if not path.endswith('.parquet'): continue
        tmp = os.path.join(OUT, '_raw.parquet')
        print('fetch', path, '%.0f MB' % (size/1e6), flush=True)
        with open(tmp, 'wb') as f:
            f.write(get(RES.format(r=repo, p=urllib.parse.quote(path))))
        tb = pq.read_table(tmp, columns=KEEP)
        parts.append(tb)
        os.remove(tmp)
        print('   rows', tb.num_rows, flush=True)
    full = pa.concat_tables(parts)
    pq.write_table(full, os.path.join(OUT, 'battles_140k.parquet'))
    print('battles total', full.num_rows, flush=True)

def leaderboard():
    repo = 'lmarena-ai/leaderboard-dataset'
    for path, size in sorted(tree(repo)):
        if path.startswith('text/') and path.endswith('.parquet'):
            local = os.path.join(OUT, 'lb_' + path.replace('/', '_'))
            with open(local, 'wb') as f:
                f.write(get(RES.format(r=repo, p=urllib.parse.quote(path))))
            print('leaderboard', path, size, flush=True)

if __name__ == '__main__':
    leaderboard()
    battles()
    print('DONE')
