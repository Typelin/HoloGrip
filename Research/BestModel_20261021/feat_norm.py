"""Which features transfer across days? Leave-one-day-out, XGBoost (GPU), TRUE hits of held-out day.
Variants: all 104 / summary 83 only / rotation 21 only / per-take z-normalised (batch: stats of the
whole take, training takes normalised the same way) / per-take centred only."""
import sys
from collections import Counter, defaultdict
import numpy as np, joblib, xgboost as xgb
import common as K
from seq_data import build
from lodo_full import day_of, DAYS

SEL = dict(all=slice(0, 104), summ=slice(0, 83), rot=slice(83, 104))


def norm(X, mode):
    if mode == 'z': return (X-X.mean(0))/(X.std(0)+1e-3)
    if mode == 'c': return X-X.mean(0)
    if mode == 'crot': Y = X.copy(); Y[:, 83:] -= X[:, 83:].mean(0); return Y
    return X


def main():
    seq, _ = build(); rows = joblib.load(K.OUT/'dataset2.joblib')['rows']
    takes = defaultdict(list)
    for r in rows: takes[(r['session'], r['hand'])].append(r)
    variants = [('all', None), ('summ', None), ('rot', None), ('all', 'z'), ('all', 'c'), ('all', 'crot')]
    res = defaultdict(Counter)
    for D in DAYS:
        for sel, mode in variants:
            for h in 'LR':
                Xs, ys = [], []
                for (s, hh), rr in takes.items():
                    if hh != h or day_of(s) == D: continue
                    X = np.stack([r['x'] for r in rr]); Xs.append(norm(X, mode)[:, SEL[sel]]); ys += [K.DRUMS.index(r['drum']) for r in rr]
                X = np.vstack(Xs); y = np.array(ys); cls = np.unique(y)
                m = xgb.XGBClassifier(n_estimators=600, max_depth=6, learning_rate=.05, subsample=.8, colsample_bytree=.6, device='cuda', tree_method='hist', random_state=0)
                m.fit(X, np.searchsorted(cls, y))
                for n, s in seq.items():
                    if s['day'] != D or h not in s['hands'] or len(s['hands'][h]['y']) < 20: continue
                    v = s['hands'][h]; pred = cls[m.predict(norm(v['x'], mode)[:, SEL[sel]])]
                    kind = 'pre' if s['test'] else 'song'; res[(sel, mode)][(D, kind)] += int((pred == v['y']).sum()); res[(sel, mode)][(D, kind, 'n')] += len(v['y'])
            c = res[(sel, mode)]
            print(D, sel, mode, ' '.join(f"{k}{c[(D, k)]/max(c[(D, k, 'n')], 1)*100:5.1f}" for k in ('pre', 'song')), flush=True)
    for v in variants:
        c = res[v]; pre = np.mean([c[(D, 'pre')]/c[(D, 'pre', 'n')] for D in DAYS]); song = np.mean([c[(D, 'song')]/c[(D, 'song', 'n')] for D in DAYS])
        print('平均', v, f'前測 {pre*100:5.1f} 歌曲 {song*100:5.1f}')


if __name__ == '__main__':
    main()
