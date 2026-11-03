"""Fetch Open LLM Leaderboard v2 per-sample details for a fixed model set.

One pinned harness run per model (the timestamp with the most task files, ties
broken by the later timestamp), so no model is a mixture of evaluations.
"""
import json, os, re, sys, urllib.request, urllib.parse, time

API   = 'https://huggingface.co/api/datasets/'
RESOLVE = 'https://huggingface.co/datasets/{repo}/resolve/main/{path}'
ROOT  = os.environ.get('CCD_ROOT', '.')
CACHE = os.path.join(ROOT, 'data_v2')
os.makedirs(CACHE, exist_ok=True)

HDRS = {'User-Agent': 'ccb/1.0'}
_tok = os.path.expanduser('~/.hf_token')
if os.path.exists(_tok):
    HDRS['Authorization'] = 'Bearer ' + open(_tok).read().strip()


MODELS = {
 'QW25' : 'Qwen__Qwen2.5-7B',
 'QW25I': 'Qwen__Qwen2.5-7B-Instruct',
 'QW2'  : 'Qwen__Qwen2-7B',
 'QW2I' : 'Qwen__Qwen2-7B-Instruct',
 'GEM2' : 'google__gemma-2-9b',
 'GEM2I': 'google__gemma-2-9b-it',
 'MI03' : 'mistralai__Mistral-7B-v0.3',
 'MI03I': 'mistralai__Mistral-7B-Instruct-v0.3',
 'LL3'  : 'meta-llama__Meta-Llama-3-8B',
 'LL3I' : 'meta-llama__Meta-Llama-3-8B-Instruct',
}
WANT = ('mmlu_pro', 'gpqa', 'musr', 'bbh', 'ifeval', 'math')

def get(url, tries=4):
    last = None
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers=HDRS)
            with urllib.request.urlopen(req, timeout=300) as r:
                return r.read()
        except Exception as e:
            last = e; time.sleep(2 + 3*k)
    raise last

def tree(repo):
    return json.loads(get(API + repo + '/tree/main?recursive=true').decode())

SAMP = re.compile(r'samples_leaderboard_(?P<task>.+)_(?P<ts>\d{4}-\d{2}-\d{2}T[\d\-.]+)\.jsonl?$')

def pick_run(files):
    runs = {}
    for path, size in files:
        m = SAMP.search(os.path.basename(path))
        if not m: continue
        runs.setdefault(m.group('ts'), {})[m.group('task')] = (path, size)
    if not runs: return None, {}
    best = sorted(runs.items(), key=lambda kv: (len(kv[1]), kv[0]))[-1]
    return best[0], best[1]

def main():
    manifest = {}
    for tag, repo_model in MODELS.items():
        repo = 'open-llm-leaderboard/' + repo_model + '-details'
        try:
            t = tree(repo)
        except Exception as e:
            manifest[tag] = {'repo': repo, 'error': repr(e)[:200]}
            print(tag, 'TREE FAIL', repr(e)[:120], flush=True); continue
        files = [(e['path'], e.get('size', 0)) for e in t if e.get('type') == 'file']
        ts, tasks = pick_run(files)
        keep = {k: v for k, v in tasks.items() if any(w in k for w in WANT)}
        man = {'repo': repo, 'timestamp': ts, 'n_tasks_in_run': len(tasks),
               'n_kept': len(keep), 'bytes': sum(v[1] for v in keep.values()), 'files': {}}
        print(tag, repo_model, 'ts=', ts, 'tasks=', len(tasks), 'keep=', len(keep),
              'MB=%.0f' % (man['bytes']/1e6), flush=True)
        for task, (path, size) in sorted(keep.items()):
            local = os.path.join(CACHE, (repo_model + '__' + os.path.basename(path)))
            if not (os.path.exists(local) and os.path.getsize(local) == size):
                b = get(RESOLVE.format(repo=repo, path=urllib.parse.quote(path)))
                with open(local, 'wb') as f: f.write(b)
            man['files'][task] = os.path.basename(local)
        for path, size in files:
            if os.path.basename(path).startswith('results_') and ts and ts in path:
                local = os.path.join(CACHE, repo_model + '__' + os.path.basename(path))
                if not os.path.exists(local):
                    with open(local, 'wb') as f:
                        f.write(get(RESOLVE.format(repo=repo, path=urllib.parse.quote(path))))
                man['results'] = os.path.basename(local)
        manifest[tag] = man
        json.dump(manifest, open(os.path.join(CACHE, 'manifest.json'), 'w'), indent=1)
    print('DONE')

if __name__ == '__main__':
    main()
