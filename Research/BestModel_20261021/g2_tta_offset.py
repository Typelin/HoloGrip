"""Goal step 2: causal test-time zero-offset adaptation (unsupervised, past hits only).

For every hit, probabilities are precomputed for a grid of drumhead-zero offsets (yaw, pitch) of that
hand. When hit i arrives, the offset is chosen from hits j < i of the same hand in this take (no labels,
no future): maximise mean log max-probability (entropy-minimisation style) minus a size penalty.
The chosen offset is then applied to hit i. Grid search over window N, penalty lam, min hits.
"""
import itertools, json
import numpy as np, joblib
import common as K, cells as Q
from phase3_aug import x_with_zero

YAW = np.arange(-30, 31, 5); PIT = np.arange(-20, 21, 5)
GRID = [(y, p) for y in YAW for p in PIT]


def precompute(com, D2, tag):
    f = K.OUT/f'g2_gridprob_{tag}.joblib'
    if f.exists(): return joblib.load(f)
    out = {}
    for k, t in D2['tests'].items():
        out[k] = {}
        for h in 'LR':
            it = sorted([c for c in t['items'] if c['hand'] == h], key=lambda c: c['t'])
            X = np.stack([x_with_zero(c, dy, dp) for dy, dp in GRID for c in it])
            p = K.proba(com, h, X).reshape(len(GRID), len(it), 7)
            out[k][h] = dict(t=np.array([c['t'] for c in it]), p=p)
        print('grid', k, flush=True)
    joblib.dump(out, f); return out


def adapt(G, N, lam, min_hits, decay=None):
    zero = GRID.index((0, 0)); off = np.array([abs(y)/30+abs(p)/20 for y, p in GRID])
    preds = {}
    for k, hh in G.items():
        P = []
        for h, d in hh.items():
            lp = np.log(d['p'].max(2)+1e-9)        # (grid, n)
            n = lp.shape[1]
            for i in range(n):
                lo = max(0, i-N)
                if i-lo >= min_hits:
                    score = lp[:, lo:i].mean(1)-lam*off; g = int(score.argmax())
                else: g = zero
                q = d['p'][g, i]; P.append(dict(t=float(d['t'][i]), hand=h, drum=K.DRUMS[int(q.argmax())]))
        preds[k] = P
    return preds


def main():
    D2 = joblib.load(K.OUT/'dataset2.joblib')
    com = K.train(D2['rows'], {}, seeds=(0, 1)); G = precompute(com, D2, 'base')
    base = Q.score_cells(adapt(G, 1, 0, 10**9)); print('no adapt', Q.fmt(base), flush=True)
    res = []
    for N, lam, mh in itertools.product([10, 20, 40, 80], [0.0, 0.02, 0.05, 0.1, 0.2], [5, 10]):
        r = Q.score_cells(adapt(G, N, lam, mh)); res.append(dict(N=N, lam=lam, min_hits=mh, min=r['min'], mean=r['mean'], overall=r['overall'], acc={f'{a}{b}': v for (a, b), v in r['acc'].items()}))
        print(N, lam, mh, Q.fmt(r), flush=True)
    (K.OUT/'g2_results.json').write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
