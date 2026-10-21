"""Goal step 4 (OFTTA-style, causal): per take and hand, keep a running mean of the zero-relative rotation
features of hits already played; shift the current hit's rotation block toward the training mean of that
hand by alpha*(train_mean - running_mean). No labels, no future hits."""
import json, itertools
import numpy as np, joblib
import common as K, cells as Q

ROT = slice(83, 104)


def main():
    D2 = joblib.load(K.OUT/'dataset2.joblib'); rows = D2['rows']
    com = K.train(rows, {}, seeds=(0, 1))
    tmean = {h: np.mean([r['x'][ROT] for r in rows if r['hand'] == h], 0) for h in 'LR'}
    res = {}
    for alpha, minh, win in itertools.product([0.0, 0.25, 0.5, 0.75, 1.0], [10, 30], [40, 100000]):
        preds = {}
        for k, t in D2['tests'].items():
            P = []
            for h in 'LR':
                it = sorted([c for c in t['items'] if c['hand'] == h], key=lambda c: c['t'])
                X = np.stack([c['x'] for c in it]).copy(); Xa = X.copy()
                for i in range(len(it)):
                    past = X[max(0, i-win):i, ROT]
                    if len(past) >= minh: Xa[i, ROT] = X[i, ROT]+alpha*(tmean[h]-past.mean(0))
                p = K.proba(com, h, Xa)
                P += [dict(t=c['t'], hand=h, drum=K.DRUMS[int(q.argmax())]) for c, q in zip(it, p)]
            preds[k] = P
        r = Q.score_cells(preds); res[f'{alpha}_{minh}_{win}'] = dict(min=r['min'], mean=r['mean'], acc={f'{a}{b}': v for (a, b), v in r['acc'].items()})
        print(alpha, minh, win, Q.fmt(r), flush=True)
    (K.OUT/'g4_results.json').write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
