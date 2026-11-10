"""Feasibility: 'personal calibration' for air drums. One take played in a known order defines this player's
drum positions (median zero-relative yaw/pitch per hand x drum, measured from that take's opening snare);
another take of the same player is then classified by the nearest position only (no model).
Poses come from the field feature extractor with the zero-direction guard off. Order keys = per-hand pause blocks.
Read-only. Compares with what the field engine (model_r10c, live) answered on the same hits."""
from __future__ import annotations
import csv, json, sys
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path
import numpy as np, joblib
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]; CODE = '11-10的包'
sys.path[:0] = [str(ROOT/CODE/'Program'), str(HERE)]
import seventh_core as core
MD = ROOT/CODE/'Data/模型實測'
S, H, C, T1, T2, R, F = '小鼓', 'Hi-Hat', 'Crash', '高音 Tom', '中音 Tom', 'Ride', '落地 Tom'
FWD = [(S, 10), (S, 20), (H, 20), (C, 20), (T1, 20), (T2, 20), (R, 20), (F, 20)]
KEYS = {  # per hand: (drum, hits) in time order; None = skip
 '161301': dict(name='人員3 空氣 正打', L=[(S, 10), (S, 20), (H, 20), (C, 19), (T1, 20), (T2, 20), (R, 20), (F, 20)], R=FWD),
 '161732': dict(name='人員3 空氣 反打', L=[(S, 10), (F, 20), (R, 20), (T2, 20), (T1, 20), (C, 22), (H, 20), (S, 20)], R=[(S, 10), (F, 24), (R, 20), (T2, 20), (T1, 20), (C, 19), (H, 20), (S, 20)]),
 '133616': dict(name='人員3 真鼓 正打', L=[(S, 10), (S, 20), (None, 1), (H, 20), (C, 20), (T1, 20), (T2, 20), (R, 20), (F, 20)], R=FWD),
 '134421': dict(name='人員3 真鼓 反打', L=[(S, 10), (F, 20), (R, 18), (T2, 20), (T1, 20), (C, 20), (H, 20), (S, 20)], R=[(S, 10), (F, 12), (R, 20), (T2, 20), (T1, 20), (C, 20), (H, 20), (S, 20)]),
}


def work(key):
    raw = next(next(p for p in MD.glob('*') if key in p.name).glob('*_raw_100hz_*.csv'))
    b = joblib.load(HERE/'out/model_r10c.joblib'); core.runtime.attach_boosters(b); b.pop('_zero_estimator', None); config = core.load_components()[1]
    hands = core.trajectory.read_raw(raw)
    with open(raw, encoding='utf-8-sig', newline='') as f: source = list(csv.DictReader(f))
    engine = core.Engine(b, config, snare_anchor=5); rt = engine.runtime; rec = []; orig = rt._classify
    def spy(hand, x):
        rec.append(core.runtime._yaw_pitch(x).tolist()); return orig(hand, x)
    rt._classify = spy; out = []
    for _, h, s in sorted((s['row_index'], h, s) for h, a in hands.items() for s in a):
        row = source[s['row_index']]; n0 = len(rec)
        p, _ = engine.push(h, s, dict(ax=s['ax'], ay=s['ay'], az=s['az'], yaw=float(row['yaw_deg']), pitch=float(row['pitch_deg']), roll=s['roll']))
        for q, yp in zip(p, rec[n0:]): out.append(dict(hand=q['hand'], t=q['peak_ms'], yaw=yp[0], pitch=yp[1], pred=q['drum']))
    hits = []
    for h in 'LR':
        Q = sorted((x for x in out if x['hand'] == h), key=lambda x: x['t']); seq = [d for d, n in KEYS[key][h] for _ in range(n)]
        if len(seq) != len(Q): print('!! count', key, h, len(seq), len(Q), flush=True)
        ref = np.median(np.array([(x['yaw'], x['pitch']) for x in Q[:10]]), 0)
        for i, (x, d) in enumerate(zip(Q, seq)):
            if d: hits.append(dict(hand=h, drum=d, opening=i < 10, pred=x['pred'], dy=(x['yaw']-ref[0]+180) % 360-180, dp=x['pitch']-ref[1]))
    return key, hits


def proto(hits):
    P = {}
    for h in 'LR':
        for d in (S, H, C, T1, T2, R, F):
            v = np.array([(x['dy'], x['dp']) for x in hits if x['hand'] == h and x['drum'] == d and not x['opening']])
            if len(v): P[(h, d)] = np.median(v, 0)
    return P


def nearest(P, x):
    c = {d: np.hypot(((x['dy']-m[0]+180) % 360-180), x['dp']-m[1]) for (h, d), m in P.items() if h == x['hand']}
    return min(c, key=c.get)


if __name__ == '__main__':
    with Pool(4) as pool: D = dict(pool.imap(work, list(KEYS)))
    for cal, test in [('161301', '161732'), ('161732', '161301'), ('133616', '134421'), ('134421', '133616'), ('133616', '161301'), ('133616', '161732')]:
        P = proto(D[cal]); T = [x for x in D[test] if not x['opening']]
        ok = sum(nearest(P, x) == x['drum'] for x in T); eng = sum(x['pred'] == x['drum'] for x in T)
        cells = defaultdict(lambda: [0, 0])
        for x in T:
            c = cells[f"{x['hand']}|{x['drum']}"]; c[1] += 1; c[0] += nearest(P, x) == x['drum']
        bad = {k: f'{v[0]}/{v[1]}' for k, v in sorted(cells.items()) if v[0] < 0.85*v[1]}
        print(f"校正用 {KEYS[cal]['name']} -> 考 {KEYS[test]['name']}: 個人位置 {ok}/{len(T)} = {100*ok/len(T):.1f}%   現場引擎 {eng}/{len(T)} = {100*eng/len(T):.1f}%   未達: {bad}")
    json.dump({k: v for k, v in D.items()}, open(HERE/'out/air_personal_1110.json', 'w', encoding='utf-8'), ensure_ascii=False)
