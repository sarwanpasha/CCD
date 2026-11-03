"""Build per-example loss matrices from Open LLM Leaderboard v2 sample dumps."""
import json, os, re, glob, sys, hashlib
import numpy as np

ROOT  = os.environ.get('CCD_ROOT', '.')
CACHE = os.path.join(ROOT, 'data_v2')
OUT   = os.path.join(ROOT, 'cache_v2.npz')

FAMILY = [('mmlu_pro', re.compile(r'samples_leaderboard_mmlu_pro_')),
          ('bbh',      re.compile(r'samples_leaderboard_bbh_')),
          ('gpqa',     re.compile(r'samples_leaderboard_gpqa_')),
          ('musr',     re.compile(r'samples_leaderboard_musr_'))]

def gold_index(rec):
    tgt = rec.get('target')
    doc = rec.get('doc', {}) or {}
    args = rec.get('arguments')
    conts = None
    if isinstance(args, dict):
        keys = sorted(args.keys(), key=lambda k: int(re.findall(r'\d+', k)[-1]))
        conts = []
        for k in keys:
            v = args[k]
            c = v.get('arg_1') if isinstance(v, dict) else (v[1] if isinstance(v, (list, tuple)) and len(v) > 1 else None)
            conts.append('' if c is None else str(c).strip())
    if conts:
        t = str(tgt).strip()
        if t in conts:
            return conts.index(t)
        m = re.fullmatch(r'\(?([A-Z])\)?', t)
        if m:
            for form in ('(%s)' % m.group(1), m.group(1)):
                if form in conts: return conts.index(form)
    if isinstance(doc.get('answer_index'), int):
        return doc['answer_index']
    if isinstance(tgt, int):
        return tgt
    return None

TEXT_FIELDS = ('question', 'Question', 'input', 'narrative', 'problem', 'prompt')

def content_key(doc):
    """hash of the item text only: stable across runs that shuffle the options."""
    if not isinstance(doc, dict): return None
    parts = [str(doc[k]) for k in TEXT_FIELDS if k in doc]
    if not parts: return None
    return hashlib.md5('||'.join(parts).encode('utf-8')).hexdigest()

def parse_file(path):
    out = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            r = json.loads(line)
            fr = r.get('filtered_resps')
            if not (isinstance(fr, list) and fr and isinstance(fr[0], (list, tuple))):
                return {}
            gi = gold_index(r)
            if gi is None or gi >= len(fr):     continue
            try:
                ll = float(fr[gi][0])
            except (TypeError, ValueError):
                continue
            acc = r.get('acc', r.get('acc_norm'))
            ck = content_key(r.get('doc'))
            out[r['doc_hash'] if ck is None else ck] = (-ll, float(acc) if acc is not None else np.nan,
                                                        r['doc_hash'])
    return out

def main():
    man = json.load(open(os.path.join(CACHE, 'manifest.json')))
    tags = [t for t, v in man.items() if 'files' in v and v['files']]
    print('models', tags, flush=True)
    store = {}
    for fam, rx in FAMILY:
        per_model = {}
        for tag in tags:
            v = man[tag]
            hit = {}
            for task, fname in v['files'].items():
                p = os.path.join(CACHE, fname)
                if not rx.search(os.path.basename(p)): continue
                d = parse_file(p)
                for k, val in d.items(): hit[k] = val
            if hit: per_model[tag] = hit
            print(fam, tag, len(hit), flush=True)
        if len(per_model) < 2: continue
        common = set.intersection(*[set(d) for d in per_model.values()])
        keys = sorted(common)
        ms = sorted(per_model)
        hsets = [set(v[2] for v in per_model[m].values()) for m in ms]
        n_hash = len(set.intersection(*hsets)) if hsets else 0
        store['nhash_' + fam] = np.array([n_hash])
        print('  content-matched %d ; harness-doc_hash-matched %d' % (len(keys), n_hash), flush=True)
        L = np.array([[per_model[m][k][0] for k in keys] for m in ms])
        A = np.array([[per_model[m][k][1] for k in keys] for m in ms])
        store['L_' + fam] = L
        store['A_' + fam] = A
        store['models_' + fam] = np.array(ms)
        store['keys_' + fam] = np.array(keys)
        print('FAMILY', fam, 'models', ms, 'n', len(keys), flush=True)
    np.savez_compressed(OUT, **store)
    print('WROTE', OUT, flush=True)

if __name__ == '__main__':
    main()
