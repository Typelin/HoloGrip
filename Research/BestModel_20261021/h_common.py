"""Exam set for the second goal run: 6 pretests (14 cells) + held-out song seventh_04 (original kit,
song 2). seventh_04 is removed from training. Adds causal timing features per hit (time since the same
hand's / other hand's previous detected peak, hits in the last 1 s) computed from detector peaks."""
import numpy as np, joblib
from collections import defaultdict
import common as K, cells as Q
import soft_hit_experiment as S

HOLD = 'seventh_04'
ROT = slice(83, 104)


def timing_table():
    f = K.OUT/'timing_peaks.joblib'
    if f.exists(): return joblib.load(f)
    C = S.C; out = {}
    for s, v in C['sessions'].items(): out[s] = sorted((c['peak']['peak_ms'], c['peak']['hand']) for c in v['items'])
    for k in K.TESTS: out[k] = sorted((c['peak']['peak_ms'], c['peak']['hand']) for c in C['tests'][k]['items'])
    joblib.dump(out, f); return out


def timing_feats(peaks, t, hand):
    prev_same = [p for p, h in peaks if h == hand and p < t-1]; prev_other = [p for p, h in peaks if h != hand and p < t-1]
    ds = t-prev_same[-1] if prev_same else 3000.; do = t-prev_other[-1] if prev_other else 3000.
    n1 = sum(1 for p, h in peaks if t-1000 <= p < t-1)
    return np.array([np.log1p(min(ds, 3000)), np.log1p(min(do, 3000)), np.log1p(n1)], np.float32)


def load(with_timing=True):
    D2 = joblib.load(K.OUT/'dataset2.joblib'); TP = timing_table() if with_timing else None
    rows = []
    for r in D2['rows']:
        r = dict(r)
        if with_timing: r['tim'] = timing_feats(TP[r['session']], r['t'], r['hand'])
        rows.append(r)
    tests = {}
    for k, t in D2['tests'].items():
        items = []
        for c in t['items']:
            c = dict(c)
            if with_timing: c['tim'] = timing_feats(TP[k], c['t'], c['hand'])
            items.append(c)
        tests[k] = dict(truth=t['truth'], items=items)
    train = [r for r in rows if r['session'] != HOLD]; hold = [r for r in rows if r['session'] == HOLD]
    return train, hold, tests


def song_acc(com, hold, fx=lambda r: r['x']):
    ok = n = 0
    for h in 'LR':
        t = [r for r in hold if r['hand'] == h]
        p = K.proba(com, h, np.stack([fx(r) for r in t]))
        ok += sum(K.DRUMS[int(q.argmax())] == r['drum'] for q, r in zip(p, t)); n += len(t)
    return ok/n


def pretest(com, tests, fx=lambda r: r['x'], post=None):
    preds = {}
    for k, t in tests.items():
        P = []
        for h in 'LR':
            it = sorted([c for c in t['items'] if c['hand'] == h], key=lambda c: c['t'])
            X = np.stack([fx(c) for c in it])
            p = post(com, h, it, X) if post else K.proba(com, h, X)
            P += [dict(t=c['t'], hand=h, drum=K.DRUMS[int(q.argmax())]) for c, q in zip(it, p)]
        preds[k] = P
    return Q.score_cells(preds)
