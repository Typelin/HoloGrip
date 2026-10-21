"""Phase 5: add other-hand context (10 dims) to x104. Same learners; pretests + group CV."""
import json, time
import numpy as np, joblib
import common as K


def main():
    D3 = joblib.load(K.OUT/'dataset3.joblib'); f = K.OUT/'phase5.jsonl'
    for name, fn in {'x104+other': lambda r: np.r_[r['x'], np.nan_to_num(r['other'], nan=0.)],
                     'x104+other_pose_only': lambda r: np.r_[r['x'], np.nan_to_num(r['other'][:7], nan=0.)]}.items():
        t0 = time.time()
        rows = [dict(r, x=fn(r).astype(np.float32)) for r in D3['rows']]
        D = dict(rows=rows, tests={k: dict(truth=v['truth'], items=[dict(c, x=fn(c).astype(np.float32)) for c in v['items']]) for k, v in D3['tests'].items()})
        com = K.train(rows, {}, seeds=(0, 1)); res, summ = K.exam(com, D); cv = K.group_cv(rows, {})
        rec = dict(name=name, pretest=summ, per={k: round(v['0.0']['correct']/v['0.0']['total'], 4) for k, v in res.items()},
                   cv={g: cv[g]['acc'] for g in ['R4', 'R5', 'R6', 'R7o', 'R7m']}, cv_mean=cv['mean_acc'], sec=round(time.time()-t0))
        with f.open('a', encoding='utf-8') as fh: fh.write(json.dumps(rec, ensure_ascii=False)+'\n')
        print(name, summ['0.0'], 'p0.6', summ['0.6'], 'cv', rec['cv_mean'], rec['cv'], rec['per'], flush=True)


if __name__ == '__main__':
    main()
