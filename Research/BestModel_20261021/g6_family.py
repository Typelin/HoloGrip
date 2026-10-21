"""Goal step 6: hierarchical decision. Family = cymbal (Hi-Hat, Crash, Ride) vs drum (snare, toms),
learned from features that do not depend on position (heading-free trajectory + impact; optional tilt).
Then the full model's probabilities are restricted to the predicted family (soft: multiply by family
probability^beta). Scored on pretest cells; family-only accuracy reported too."""
import json, itertools
import numpy as np, joblib
import xgboost as xgb
import common as K, cells as Q
from g3_features import tilt

CYM = {'Hi-Hat', 'Crash', 'Ride'}


def fam_feats(r, mode):
    if mode == 'x83': return r['x'][:83]
    if mode == 'hf': return r['hf']
    if mode == 'hf+tilt': return np.r_[r['hf'], tilt(r)]
    if mode == 'x83+tilt': return np.r_[r['x'][:83], tilt(r)]


def main():
    D2 = joblib.load(K.OUT/'dataset2.joblib'); rows = D2['rows']; T = Q.truth_hands()
    com = K.train(rows, {}, seeds=(0, 1))
    base = {k: {h: K.proba(com, h, np.stack([c['x'] for c in t['items'] if c['hand'] == h])) for h in 'LR'} for k, t in D2['tests'].items()}
    out = {}
    for mode in ['x83', 'hf', 'x83+tilt', 'hf+tilt']:
        fam = {}
        for h in 'LR':
            rr = [r for r in rows if r['hand'] == h]; X = np.stack([fam_feats(r, mode) for r in rr]); y = np.array([r['drum'] in CYM for r in rr], int)
            ms = []
            for s in range(2):
                m = xgb.XGBClassifier(n_estimators=800, max_depth=5, learning_rate=.05, subsample=.8, colsample_bytree=.6, device='cuda', tree_method='hist', random_state=s)
                w = np.where(y == 1, len(y)/(2*y.sum()), len(y)/(2*(len(y)-y.sum()))); m.fit(X, y, sample_weight=w); ms.append(m)
            fam[h] = {k: np.mean([m.predict_proba(np.stack([fam_feats(c, mode) for c in t['items'] if c['hand'] == h]))[:, 1] for m in ms], 0) for k, t in D2['tests'].items()}
        for beta in [0, 1, 2, 4]:
            preds = {}
            for k, t in D2['tests'].items():
                P = []
                for h in 'LR':
                    it = [c for c in t['items'] if c['hand'] == h]; p = base[k][h].copy(); pc = fam[h][k][:, None]
                    mask = np.array([d in CYM for d in K.DRUMS])
                    p = p*np.where(mask, pc, 1-pc)**beta
                    P += [dict(t=c['t'], hand=h, drum=K.DRUMS[int(q.argmax())]) for c, q in zip(it, p)]
                preds[k] = P
            r = Q.score_cells(preds); out[f'{mode}_b{beta}'] = dict(min=r['min'], mean=r['mean'], acc={f'{a}{b}': v for (a, b), v in r['acc'].items()})
            print(mode, beta, Q.fmt(r), flush=True)
    (K.OUT/'g6_results.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
