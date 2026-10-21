"""Goal step 7: songs play Ride from a high arm (pitch ~60) but pretests from a low arm (~35).
Train with pitch-only zero jitter (model relies less on stroke height, more on direction), then apply
the causal re-centering of step 4. Grid over pitch jitter and alpha."""
import json, itertools
import numpy as np, joblib
import common as K, cells as Q
from phase3_aug import x_with_zero

ROT = slice(83, 104)


def aug_pitch(rows, ap, yaw, copies, seed=0):
    if ap == 0 and yaw == 0: return rows
    rng = np.random.default_rng(seed); out = list(rows)
    for r in rows:
        for _ in range(copies): out.append(dict(r, x=x_with_zero(r, rng.uniform(-yaw, yaw), rng.uniform(-ap, ap))))
    return out


def recentered_preds(com, D2, tmean, alpha, minh=10):
    preds = {}
    for k, t in D2['tests'].items():
        P = []
        for h in 'LR':
            it = sorted([c for c in t['items'] if c['hand'] == h], key=lambda c: c['t'])
            X = np.stack([c['x'] for c in it]); Xa = X.copy()
            for i in range(len(it)):
                if i >= minh: Xa[i, ROT] = X[i, ROT]+alpha*(tmean[h]-X[:i, ROT].mean(0))
            p = K.proba(com, h, Xa); P += [dict(t=c['t'], hand=h, drum=K.DRUMS[int(q.argmax())], score=float(q.max())) for c, q in zip(it, p)]
        preds[k] = P
    return preds


def main():
    D2 = joblib.load(K.OUT/'dataset2.joblib'); rows = D2['rows']; res = {}
    tmean = {h: np.mean([r['x'][ROT] for r in rows if r['hand'] == h], 0) for h in 'LR'}
    for ap, yaw in [(0, 0), (15, 0), (25, 0), (25, 5)]:
        com = K.train(aug_pitch(rows, ap, yaw, 3), {}, seeds=(0, 1))
        for alpha in [0.0, 0.25, 0.4]:
            preds = recentered_preds(com, D2, tmean, alpha); r = Q.score_cells(preds)
            res[f'p{ap}_y{yaw}_a{alpha}'] = dict(min=r['min'], mean=r['mean'], acc={f'{a}{b}': v for (a, b), v in r['acc'].items()})
            joblib.dump(preds, K.OUT/f'g7_preds_p{ap}_y{yaw}_a{alpha}.joblib')
            print(ap, yaw, alpha, Q.fmt(r), flush=True)
    (K.OUT/'g7_results.json').write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
