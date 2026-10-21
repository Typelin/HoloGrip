"""Phase 2: feature sets (same learners/weights as base_ens). Results to out/phase2.jsonl."""
import json, time
import numpy as np, joblib
import common as K

SETS = {
    'x104': lambda r: r['x'],
    'x104+dense': lambda r: np.r_[r['x'], r['dense']],
    'hf516+dense': lambda r: np.r_[r['hf'], r['dense']],
    'hf516+x104': lambda r: np.r_[r['hf'], r['x']],
    'all': lambda r: np.r_[r['hf'], r['x'], r['dense']],
    'x83+dense': lambda r: np.r_[r['x'][:83], r['dense']],
}


def main():
    D2 = joblib.load(K.OUT/'dataset2.joblib')
    f = K.OUT/'phase2.jsonl'; done = set()
    if f.exists(): done = {json.loads(l)['name'] for l in f.read_text(encoding='utf-8').splitlines() if l.strip()}
    for name, fn in SETS.items():
        if name in done: continue
        t0 = time.time()
        rows = [dict(r, x=fn(r).astype(np.float32)) for r in D2['rows']]
        D = dict(rows=rows, tests={k: dict(truth=v['truth'], items=[dict(c, x=fn(c).astype(np.float32)) for c in v['items']]) for k, v in D2['tests'].items()})
        com = K.train(rows, {}, seeds=(0, 1)); res, summ = K.exam(com, D); cv = K.group_cv(rows, {})
        rec = dict(name=name, dims=int(rows[0]['x'].shape[0]), pretest=summ, per={k: v['0.0'] for k, v in res.items()}, cv=cv, sec=round(time.time()-t0))
        with f.open('a', encoding='utf-8') as fh: fh.write(json.dumps(rec, ensure_ascii=False, default=float)+'\n')
        print(name, rec['dims'], summ['0.0'], summ['0.6'], 'cv', cv['mean_acc'], rec['sec'], flush=True)


if __name__ == '__main__':
    main()
