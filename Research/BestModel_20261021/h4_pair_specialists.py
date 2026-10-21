"""Pairwise specialists for the confused neighbours (per hand, balanced, rich features: x104 + dense
rotation path + absolute tilt). When the base model's top-2 is such a pair, the specialist decides
between the two. Checked cross-day (group out) for each pair, then on pretests and held-out song."""
import json
import numpy as np, joblib
import xgboost as xgb
import common as K, cells as Q, h_common as H
from g3_features import tilt
from h2_style_offset import train_w

PAIRS = [('R', 'Crash', 'Hi-Hat'), ('R', 'Ride', '落地 Tom'), ('R', 'Ride', '中音 Tom'), ('R', '高音 Tom', 'Hi-Hat'), ('R', '高音 Tom', '小鼓'),
         ('L', '中音 Tom', '高音 Tom'), ('L', '落地 Tom', '中音 Tom'), ('L', 'Ride', '落地 Tom'), ('R', '小鼓', 'Hi-Hat')]


def rich(r): return np.r_[r['x'], r['dense'], tilt(r)].astype(np.float32)


def fit_pair(rows, h, a, b, seed=0):
    rr = [r for r in rows if r['hand'] == h and r['drum'] in (a, b)]
    if len(rr) < 40 or len({r['drum'] for r in rr}) < 2: return None
    X = np.stack([rich(r) for r in rr]); y = np.array([r['drum'] == a for r in rr], int)
    w = np.where(y == 1, len(y)/(2*y.sum()), len(y)/(2*(len(y)-y.sum())))
    m = xgb.XGBClassifier(n_estimators=500, max_depth=4, learning_rate=.05, subsample=.8, colsample_bytree=.5, device='cuda', tree_method='hist', random_state=seed)
    m.fit(X, y, sample_weight=w); return m


def main():
    train, hold, tests = H.load(with_timing=False); out = {}
    for h, a, b in PAIRS:   # cross-day check of each specialist
        accs = []
        for g in ['R4', 'R5', 'R6', 'R7o', 'R7m']:
            te = [r for r in train if r['group'] == g and r['hand'] == h and r['drum'] in (a, b)]
            m = fit_pair([r for r in train if r['group'] != g], h, a, b)
            if m is None or len(te) < 10: continue
            p = m.predict_proba(np.stack([rich(r) for r in te]))[:, 1]; accs.append((g, round(float(np.mean((p > .5) == np.array([r['drum'] == a for r in te]))), 3), len(te)))
        out[f'{h}{a}|{b}'] = accs; print('pair', h, a, b, accs, flush=True)
    com = train_w(train, lambda r: r['x'], lambda r: 1.)
    spec = {(h, a, b): fit_pair(train, h, a, b) for h, a, b in PAIRS}

    def post(com_, h, it, X):
        p = K.proba(com_, h, X).copy()
        R = np.stack([rich(c) for c in it])
        for (hh, a, b), m in spec.items():
            if hh != h or m is None: continue
            ia, ib = K.DRUMS.index(a), K.DRUMS.index(b); top2 = np.argsort(-p, 1)[:, :2]
            sel = np.array([set(t) == {ia, ib} for t in top2])
            if sel.any():
                q = m.predict_proba(R[sel])[:, 1]; s = p[sel, ia]+p[sel, ib]
                p[sel, ia] = s*q; p[sel, ib] = s*(1-q)
        return p
    r0 = H.pretest(com, tests); r1 = H.pretest(com, tests, post=post)
    print('base ', Q.fmt(r0)); print('pairs', Q.fmt(r1))
    out['pretest'] = dict(base={f'{a}{b}': v for (a, b), v in r0['acc'].items()}, pairs={f'{a}{b}': v for (a, b), v in r1['acc'].items()})
    (K.OUT/'h4.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
