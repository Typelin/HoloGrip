"""(a) Timing features (causal tempo context) and style weighting (single-hand sessions / isolated hits).
(b) Prototype-aligned per-take offset: from earlier confident hits of the same hand, compare the mean
rotation per PREDICTED drum with that drum's training mean; the count-weighted mean gap shifts the
current hit (base predictions are used, no iteration; past hits only). Works for single-drum takes.
Exam: 14 pretest cells + held-out song 04."""
import json, itertools
import numpy as np
import common as K, cells as Q, h_common as H

ROT = H.ROT
SINGLES = {'sixth_S04', 'sixth_S05', 'sixth_S06', 'sixth_S07'} | {f'seventh_{i:02d}' for i in range(6, 12)}


def train_w(train, fx, wfun, seeds=(0, 1)):
    rows = [dict(r, x=fx(r)) for r in train]
    cfg = dict(group_w={})
    com = []
    for s in seeds:
        member = {}
        for h in 'LR':
            rr = [r for r in rows if r['hand'] == h]
            from collections import Counter
            c = Counter(r['drum'] for r in rr); w = np.array([wfun(r)/np.sqrt(c[r['drum']]) for r in rr], np.float32); w /= w.mean()
            X = np.stack([r['x'] for r in rr]); y = [r['drum'] for r in rr]; classes = sorted(set(y))
            import xgboost as xgb
            m = xgb.XGBClassifier(n_estimators=600, max_depth=6, learning_rate=.05, subsample=.8, colsample_bytree=.6, reg_lambda=2., device='cuda', tree_method='hist', random_state=s)
            m.fit(X, np.array([classes.index(v) for v in y]), sample_weight=w)
            member[h] = (classes, [m, K.MLP(classes, seed=s).fit(X, y, w)])
        com.append(member)
    return com


def proto_post(train, alpha, conf=0.7, minh=10):
    tm = {(h, d): np.mean([r['x'][ROT] for r in train if r['hand'] == h and r['drum'] == d], 0) for h in 'LR' for d in K.DRUMS
          if any(r['hand'] == h and r['drum'] == d for r in train)}

    def post(com, h, it, X):
        p0 = K.proba(com, h, X); pred = p0.argmax(1); Xa = X.copy()
        for i in range(len(X)):
            past = [j for j in range(i) if p0[j].max() >= conf]
            if len(past) < minh: continue
            gaps, ws = [], []
            for d in set(pred[past]):
                js = [j for j in past if pred[j] == d]
                key = (h, K.DRUMS[d])
                if key in tm: gaps.append(tm[key]-X[js, ROT].mean(0)); ws.append(len(js))
            if gaps: Xa[i, ROT] = X[i, ROT]+alpha*np.average(np.array(gaps), axis=0, weights=ws)
        return K.proba(com, h, Xa)
    return post


def main():
    train, hold, tests = H.load(with_timing=True); out = {}
    fx_base = lambda r: r['x']; fx_tim = lambda r: np.r_[r['x'], r['tim']]
    exps = {
        'base': (fx_base, lambda r: 1.),
        'timing': (fx_tim, lambda r: 1.),
        'timing_singles_x2': (fx_tim, lambda r: 2. if r['session'] in SINGLES else 1.),
        'timing_isolated_x3': (fx_tim, lambda r: 3. if r['tim'][0] > np.log1p(400) else 1.),
        'base_singles_x2': (fx_base, lambda r: 2. if r['session'] in SINGLES else 1.),
    }
    for name, (fx, wf) in exps.items():
        com = train_w(train, fx, wf)
        for alpha in (0.0, 0.5, 1.0):
            post = None if alpha == 0 else proto_post([dict(r, x=fx(r)) for r in train], alpha)
            r = H.pretest(com, tests, fx, post); s = H.song_acc(com, hold, fx)
            out[f'{name}_a{alpha}'] = dict(min=r['min'], mean=r['mean'], song=s, acc={f'{a}{b}': v for (a, b), v in r['acc'].items()})
            print(name, alpha, Q.fmt(r), 'song04', round(s, 4), flush=True)
        (K.OUT/'h2.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
