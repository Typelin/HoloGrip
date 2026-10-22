"""Layout augmentation: copies of every training take with a random take-wide shift of the rotation block
(same 3-vector added at all 7 time points) and a small random per-drum jitter, to imitate another day.
Leave-one-day-out XGBoost (GPU), raw output on TRUE hits of the held-out day (no runtime correction)."""
from collections import Counter, defaultdict
import numpy as np, joblib, xgboost as xgb
import common as K
from seq_data import build
from lodo_full import day_of, DAYS


def augmented(takes, copies, s_sd, j_sd, rng):
    Xs, ys = [], []
    for rr in takes:
        X = np.stack([r['x'] for r in rr]); y = np.array([K.DRUMS.index(r['drum']) for r in rr]); Xs.append(X); ys.append(y)
        for _ in range(copies):
            Z = X.copy(); s = rng.normal(0, s_sd, 3); jit = rng.normal(0, j_sd, (7, 3))
            Z[:, 83:104] += np.tile(s, 7)+np.tile(jit[y], (1, 7)); Xs.append(Z); ys.append(y)
    return np.vstack(Xs), np.concatenate(ys)


def main():
    seq, _ = build(); rows = joblib.load(K.OUT/'dataset2.joblib')['rows']; takes = defaultdict(list)
    for r in rows: takes[(r['session'], r['hand'])].append(r)
    variants = [(0, 0, 0), (3, 0.05, 0.0), (3, 0.1, 0.0), (3, 0.1, 0.03), (3, 0.2, 0.05)]
    res = defaultdict(Counter)
    for D in DAYS:
        for va in variants:
            rng = np.random.default_rng(0)
            for h in 'LR':
                X, y = augmented([rr for (s, hh), rr in takes.items() if hh == h and day_of(s) != D], *va, rng); cls = np.unique(y)
                m = xgb.XGBClassifier(n_estimators=600, max_depth=6, learning_rate=.05, subsample=.8, colsample_bytree=.6, device='cuda', tree_method='hist', random_state=0).fit(X, np.searchsorted(cls, y))
                for n, s in seq.items():
                    if s['day'] != D or h not in s['hands'] or len(s['hands'][h]['y']) < 20: continue
                    v = s['hands'][h]; kind = 'pre' if s['test'] else 'song'
                    res[va][(D, kind)] += int((cls[m.predict(v['x'])] == v['y']).sum()); res[va][(D, kind, 'n')] += len(v['y'])
            c = res[va]; print(D, va, ' '.join(f"{k} {c[(D, k)]/max(c[(D, k, 'n')], 1)*100:5.1f}" for k in ('pre', 'song')), flush=True)
    for va in variants:
        c = res[va]; print('平均', va, f"前測 {np.mean([c[(D, 'pre')]/c[(D, 'pre', 'n')] for D in DAYS])*100:5.1f} 歌曲 {np.mean([c[(D, 'song')]/c[(D, 'song', 'n')] for D in DAYS])*100:5.1f}")


if __name__ == '__main__':
    main()
