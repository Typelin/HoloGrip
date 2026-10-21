"""Pretest 14-cell exam. Hand answer per MIDI = glove with the clear peak (pretests are one-hand
deliberate hits). Cached once. score_cells(pred_fn) -> per-cell accuracy, min, mean, overall."""
from collections import Counter, defaultdict
import numpy as np, joblib
from scipy.optimize import linear_sum_assignment
import common as K

CELLS = [(h, d) for h in 'LR' for d in K.DRUMS]


def truth_hands():
    f = K.OUT/'pretest_truth_hands.joblib'
    if f.exists(): return joblib.load(f)
    import soft_hit_experiment as S, venue_free_runtime_1020 as rt
    out = {}
    for k in K.TESTS:
        hands = rt.world.read_raw(S.C['tests'][k]['raw'])
        ser = {h: (np.array([r['t'] for r in a]), np.array([abs(r['mag']-1) for r in a])) for h, a in hands.items()}
        def act(h, t0):
            ts, ac = ser[h]; i, j = np.searchsorted(ts, t0-60), np.searchsorted(ts, t0+60, 'right'); return float(ac[i:j].max()) if j > i else 0.
        out[k] = [dict(t=float(g['t']), drum=g['drum'], hand=None if max(act('L', g['t']), act('R', g['t'])) < .3 else ('L' if act('L', g['t']) > act('R', g['t']) else 'R'))
                  for g in S.C['tests'][k]['truth']]
    joblib.dump(out, f); return out


TRUTH = None


def score_cells(preds_by_test):
    """preds_by_test: {test: [dict(t, hand, drum)]}"""
    global TRUTH
    if TRUTH is None: TRUTH = truth_hands()
    cells = defaultdict(Counter); total = ok_all = 0
    for k, truth in TRUTH.items():
        P = preds_by_test[k]; tt = np.array([g['t'] for g in truth]); pt = np.array([p['t'] for p in P])
        dt = np.abs(tt[:, None]-pt[None, :]); hm = np.array([[g['hand'] is not None and g['hand'] != p['hand'] for p in P] for g in truth])
        cost = np.hstack([np.where(dt <= 80, dt+300*hm, 1e9), np.full((len(tt), len(tt)), 1e6)])
        r, c = linear_sum_assignment(cost); pairs = {i: j for i, j in zip(r, c) if j < len(pt) and cost[i, j] <= 380}
        for i, g in enumerate(truth):
            j = pairs.get(i)
            o = 'miss' if j is None else ('hand' if g['hand'] and P[j]['hand'] != g['hand'] else P[j]['drum'])
            if g['hand']: cells[(g['hand'], g['drum'])][o] += 1
            total += 1; ok_all += (o == g['drum'])
    acc = {c: cells[c][c[1]]/max(sum(cells[c].values()), 1) for c in CELLS}
    return dict(acc=acc, min=min(acc.values()), mean=float(np.mean(list(acc.values()))), overall=ok_all/total,
                confusion={c: Counter({k: v for k, v in cells[c].items() if k != c[1]}).most_common(2) for c in CELLS})


def preds_from_proba(tests, proba_fn):
    """proba_fn(hand, items) -> (n,7) probabilities in K.DRUMS order."""
    out = {}
    for k, t in tests.items():
        P = []
        for h in 'LR':
            it = [c for c in t['items'] if c['hand'] == h]
            p = proba_fn(h, it)
            P += [dict(t=c['t'], hand=h, drum=K.DRUMS[int(q.argmax())], p=q) for c, q in zip(it, p)]
        out[k] = P
    return out


def fmt(res):
    return ' '.join(f"{h}{d[:2]}{res['acc'][(h, d)]*100:.0f}" for h, d in CELLS)+f" | min {res['min']*100:.1f} mean {res['mean']*100:.1f} all {res['overall']*100:.1f}"
