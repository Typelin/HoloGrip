"""Per recording and hand: time-ordered TRUE hits (1.5 g candidates matched to labelled MIDI, same hand,
<= 60 ms) with x (104), drum, t. Plus leave-one-day-out committees (trained without that day) cached
with drum rotation means. Used to study cross-day drum classification apart from the hit gate."""
import sys
import numpy as np, joblib
from scipy.optimize import linear_sum_assignment
sys.argv = ['x', '1p5']
import hit_clf as H, common as K, full_report as F
from verify_1023 import TRUTH
from lodo_full import day_of, DAYS

SEQ_F = K.OUT/'seq_true_hits.joblib'; COM_F = K.OUT/'lodo_committees.joblib'


def true_hits(name):
    used = [g for g in TRUTH[name] if g['hand']] if name in K.TESTS else F.truth_of(name)[0]
    out = {}
    for h in 'LR':
        cs = sorted([c for c in H.CAND[name] if c['hand'] == h], key=lambda c: c['t']); us = [u for u in used if u['hand'] == h]
        if not cs or not us: continue
        dt = np.abs(np.array([c['t'] for c in cs])[:, None]-np.array([u['t'] for u in us])[None, :]); cost = np.where(dt <= 60, dt, 1e9)
        r, c = linear_sum_assignment(cost); keep = sorted((cs[a]['t'], a, b) for a, b in zip(r, c) if cost[a, b] <= 60)
        if not keep: continue
        out[h] = dict(t=np.array([k[0] for k in keep]), x=np.stack([cs[k[1]]['x'] for k in keep]).astype(np.float32),
                      y=np.array([K.DRUMS.index(us[k[2]]['drum']) for k in keep]))
    return out


def build():
    if SEQ_F.exists(): seq = joblib.load(SEQ_F)
    else:
        seq = {n: dict(day=day_of(n), test=n in K.TESTS, hands=true_hits(n)) for n in H.CAND}; joblib.dump(seq, SEQ_F)
    if COM_F.exists(): return seq, joblib.load(COM_F)
    rows = joblib.load(K.OUT/'dataset2.joblib')['rows']; coms = {}
    for D in DAYS:
        tr = [r for r in rows if day_of(r['session']) != D]
        tm = {h: np.array([np.mean([r['x'][83:104] for r in tr if r['hand'] == h and r['drum'] == d], 0) if any(r['hand'] == h and r['drum'] == d for r in tr) else np.zeros(21) for d in K.DRUMS]) for h in 'LR'}
        coms[D] = dict(com=K.train(tr, {}, seeds=(0, 1)), tm=tm); print('trained', D, flush=True)
    joblib.dump(coms, COM_F); return seq, coms


if __name__ == '__main__':
    seq, coms = build()
    for n, s in seq.items(): print(n, s['day'], {h: len(v['y']) for h, v in s['hands'].items()})
