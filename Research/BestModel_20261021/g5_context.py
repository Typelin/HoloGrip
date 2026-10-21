"""Goal step 5: take-context model (causal). Each hit gets the running mean of the zero-relative rotation
block of the SAME hand's earlier hits in the take (+ log count). Training adds per-session random zero
offsets applied consistently to the hit and its context, so the model learns position relative to
context. Test context is built from earlier detected hits only (no labels, no future)."""
import json, itertools
import numpy as np, joblib
from collections import defaultdict
import common as K, cells as Q
from phase3_aug import x_with_zero

ROT = slice(83, 104)


def with_context(seq_x, seq_rot, minh=5, prior=None):
    """seq sorted by time; returns features x + ctx + (rot-ctx) + logn."""
    out = []; s = np.zeros(21); n = 0
    for x, rot in zip(seq_x, seq_rot):
        ctx = s/n if n >= minh else prior
        out.append(np.r_[x, ctx, rot-ctx, np.log1p(n)].astype(np.float32)); s = s+rot; n += 1
    return out


def build_train(rows, copies, a, seed=0):
    rng = np.random.default_rng(seed); by = defaultdict(list)
    for r in rows: by[(r['session'], r['hand'])].append(r)
    prior = {h: np.mean([r['x'][ROT] for r in rows if r['hand'] == h], 0) for h in 'LR'}
    out = []
    for (s, h), rs in by.items():
        rs = sorted(rs, key=lambda r: r['t'])
        for c in range(copies+1):
            dy, dp = (0., 0.) if c == 0 else (rng.uniform(-a, a), rng.uniform(-a/2, a/2))
            xs = [x_with_zero(r, dy, dp) for r in rs]
            feats = with_context(xs, [x[ROT] for x in xs], prior=prior[h])
            out += [dict(r, x=f) for r, f in zip(rs, feats)]
    return out, prior


def test_items(D2, prior):
    tests = {}
    for k, t in D2['tests'].items():
        items = []
        for h in 'LR':
            it = sorted([c for c in t['items'] if c['hand'] == h], key=lambda c: c['t'])
            feats = with_context([c['x'] for c in it], [c['x'][ROT] for c in it], prior=prior[h])
            items += [dict(c, x=f) for c, f in zip(it, feats)]
        tests[k] = dict(truth=t['truth'], items=items)
    return tests


def main():
    D2 = joblib.load(K.OUT/'dataset2.joblib'); res = {}
    for copies, a in [(0, 0), (3, 10), (3, 20), (6, 20), (6, 30)]:
        tr, prior = build_train(D2['rows'], copies, a)
        com = K.train(tr, {}, seeds=(0, 1)); tests = test_items(D2, prior)
        preds = Q.preds_from_proba(tests, lambda h, it: K.proba(com, h, np.stack([c['x'] for c in it])))
        r = Q.score_cells(preds); res[f'c{copies}_a{a}'] = dict(min=r['min'], mean=r['mean'], acc={f'{x}{y}': v for (x, y), v in r['acc'].items()})
        joblib.dump(preds, K.OUT/f'g5_preds_c{copies}_a{a}.joblib')
        print(copies, a, Q.fmt(r), flush=True)
    (K.OUT/'g5_results.json').write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
