"""Leave-one-day-out: snare-anchor calibration for pretest mode.
Protocol (already used in pretests round 5/6/7): each hand first taps the snare 10 times. Variant F seeds
the per-drum offset statistics with each hand's first N accepted hits labelled as snare (they are
known by protocol, not by MIDI), then continues as variant E. Round 4's pretest started with Hi-Hat,
so it is reported separately (anchor not applicable)."""
import json, sys
from collections import Counter, defaultdict
import numpy as np, joblib, xgboost as xgb
import lodo_full as LF
H, K, F = LF.H, LF.K, LF.F
from verify_1023 import TRUTH, match

SNARE = K.DRUMS.index('小鼓')


def offsets_seeded(X, p, tm, n_anchor, conf=.7, minh=20, min_anchor=5):
    n = len(X); pred = p.argmax(1); ok = p.max(1) >= conf
    sums = np.zeros((7, 21)); cnt = np.zeros(7); out = np.zeros((n, 21)); valid = np.zeros(n, bool)
    for i in range(n):
        m = cnt > 0
        if (cnt[m].sum() >= minh) or (n_anchor and cnt[SNARE] >= min_anchor and i >= min_anchor):
            out[i] = (cnt[m, None]*tm[m]-sums[m]).sum(0)/cnt[m].sum(); valid[i] = True
        if i < n_anchor: sums[SNARE] += X[i, 83:104]; cnt[SNARE] += 1
        elif ok[i]: sums[pred[i]] += X[i, 83:104]; cnt[pred[i]] += 1
    return out, valid


def run(name, com, tm, gate, n_anchor, min_anchor):
    P = []
    for h in 'LR':
        cs = sorted([c for c in H.CAND[name] if c['hand'] == h], key=lambda c: c['t'])
        ph = gate.predict_proba(np.stack([H.feats(c) for c in cs]))[:, 1]; acc = [c for c, q in zip(cs, ph) if q >= .4]
        X = np.stack([c['x'] for c in acc]).astype(np.float32); p0 = K.proba(com, h, X)
        o1, v1 = offsets_seeded(X, p0, tm[h], n_anchor, min_anchor=min_anchor); X1 = X.copy(); X1[:, 83:104] += o1; p1 = np.where(v1[:, None], K.proba(com, h, X1), p0)
        o2, v2 = offsets_seeded(X, p1, tm[h], n_anchor, min_anchor=min_anchor); X2 = X.copy(); X2[:, 83:104] += o2; p = np.where(v2[:, None], K.proba(com, h, X2), p1)
        lab = p.argmax(1).copy(); sc = p.max(1); Y = LF.yp(X2); t = np.array([c['t'] for c in acc])
        for i in range(1, len(lab)):
            if sc[i] < .9 and t[i]-t[i-1] <= 600 and np.hypot(*(Y[i]-Y[i-1])) <= 16 and lab[i-1] in np.argsort(-p[i])[:2]: lab[i] = lab[i-1]
        P += [dict(t=ti, peak_ms=ti, hand=h, drum=K.DRUMS[int(l)]) for ti, l in zip(t, lab)]
    return P


def main():
    D2 = joblib.load(K.OUT/'dataset2.joblib'); rows = D2['rows']; names = list(H.CAND); L = {n: H.labels(n) for n in names}
    out = {}
    for D in LF.DAYS:
        tr = [r for r in rows if LF.day_of(r['session']) != D]; com = K.train(tr, {}, seeds=(0, 1))
        tm = {h: np.array([np.mean([r['x'][83:104] for r in tr if r['hand'] == h and r['drum'] == d], 0) if any(r['hand'] == h and r['drum'] == d for r in tr) else np.zeros(21) for d in K.DRUMS]) for h in 'LR'}
        gn = [n for n in names if n not in K.TESTS and LF.day_of(n) != D]
        Xg = np.vstack([np.stack([H.feats(c) for c in H.CAND[n]])[L[n][0] != 2] for n in gn]); yg = np.concatenate([L[n][0][L[n][0] != 2] for n in gn])
        gate = xgb.XGBClassifier(n_estimators=800, max_depth=6, learning_rate=.05, subsample=.8, colsample_bytree=.6, device='cuda', tree_method='hist', random_state=0).fit(Xg, yg)
        for na, ma in [(0, 5), (5, 5), (10, 5), (10, 10)]:
            cells = defaultdict(Counter); tot = Counter()
            for n in [x for x in K.TESTS if LF.day_of(x) == D]:
                P = run(n, com, tm, gate, na, ma); truth = TRUTH[n]; pairs = match(truth, P)
                for i, g in enumerate(truth):
                    j = pairs.get(i); ok = j is not None and (g['hand'] is None or P[j]['hand'] == g['hand']) and P[j]['drum'] == g['drum']
                    tot['ok'] += ok; tot['n'] += 1
                    if g['hand']: cells[(g['hand'], g['drum'])]['ok'] += ok; cells[(g['hand'], g['drum'])]['n'] += 1
            acc = {f'{h}{d}': round(v['ok']/v['n'], 2) for (h, d), v in cells.items()}
            out[f'{D}_a{na}_m{ma}'] = dict(pretest=round(tot['ok']/tot['n'], 4), cells=acc)
            print(D, f'小鼓錨點 {na} 下(最少 {ma})', '| 前測', round(tot['ok']/tot['n'], 4), '| 14格平均', round(float(np.mean(list(acc.values()))), 4), flush=True)
    (K.OUT/'lodo_anchor.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
