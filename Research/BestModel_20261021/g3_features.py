"""Goal step 3: zero-free features. Absolute pitch/roll (gravity referenced) along the stroke do not depend
on the drumhead zero or the room heading. Compare feature sets on the 14 pretest cells + cross-day CV."""
import json
import numpy as np, joblib
import common as K, cells as Q


def tilt(r):
    p = np.asarray(r['pose'])  # 15 x (abs yaw, abs pitch, roll)
    return np.r_[p[:, 1]/90, np.sin(np.radians(p[:, 2])), np.cos(np.radians(p[:, 2]))].astype(np.float32)


SETS = {
    'x104': lambda r: r['x'],
    'x83+tilt': lambda r: np.r_[r['x'][:83], tilt(r)],
    'x104+tilt': lambda r: np.r_[r['x'], tilt(r)],
    'x83+tilt+dense': lambda r: np.r_[r['x'][:83], tilt(r), r['dense']],
    'hf516+tilt': lambda r: np.r_[r['hf'], tilt(r)],
}


def main():
    D2 = joblib.load(K.OUT/'dataset2.joblib'); out = {}
    for name, fn in SETS.items():
        rows = [dict(r, x=fn(r)) for r in D2['rows']]
        tests = {k: dict(truth=v['truth'], items=[dict(c, x=fn(c)) for c in v['items']]) for k, v in D2['tests'].items()}
        com = K.train(rows, {}, seeds=(0, 1))
        preds = Q.preds_from_proba(tests, lambda h, it: K.proba(com, h, np.stack([c['x'] for c in it])))
        r = Q.score_cells(preds); cv = K.group_cv(rows, {})
        out[name] = dict(min=r['min'], mean=r['mean'], overall=r['overall'], cv=cv['mean_acc'], acc={f'{a}{b}': v for (a, b), v in r['acc'].items()})
        joblib.dump(preds, K.OUT/f'g3_preds_{name}.joblib')
        print(name, Q.fmt(r), 'cv', cv['mean_acc'], flush=True)
    (K.OUT/'g3_results.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
