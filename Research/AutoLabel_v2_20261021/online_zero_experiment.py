"""Causal online zero correction (strict standard: only hits already judged, never future data).

Idea: the drumhead zero gesture differs per take, so every hit's zero-relative pose is shifted by
the same yaw/pitch error. After each confident prediction (score >= CONF), compare that hit's
relative yaw/pitch at the peak with the training centre of the predicted (hand, drum). The median
gap of the last N confident hits (clamped) is applied to the zero for the NEXT hits only.
Same model for both modes; pretests and song 14 never trained.
"""
import json, numpy as np
from collections import defaultdict, deque
from pathlib import Path
from scipy.spatial.transform import Rotation as Rot
from scipy.optimize import linear_sum_assignment
import soft_hit_experiment as S, weight_experiment as W
import venue_free_runtime_1020 as rt, causal_hit_detector_1016 as detector, seventh_core as core

CONF, N, CLAMP = 0.9, 15, 25.0


def rel_yp(x):
    v = np.asarray(x[-21:]).reshape(7, 3)[4]*np.pi
    e = Rot.from_rotvec(v).as_euler('ZYX', degrees=True); return e[0], e[1]


def main():
    lab = json.loads(S.LAB13.read_text(encoding='utf-8'))['labels']; r13, _ = S.rows_from_labels('seventh_13', lab)
    rows = [dict(hand=r['hand'], drum=r['drum'], x=np.asarray(r['x'], np.float32)) for r in S.C['baseline']]
    rows += [dict(hand=r['hand'], drum=r['drum'], x=np.asarray(r['x'], np.float32)) for r in S.C['auto']
             if r['session'].startswith('seventh_') and r['session'] not in ('seventh_13', 'seventh_14')]
    rows += [dict(hand=r['hand'], drum=r['drum'], x=r['x']) for r in r13]
    for q in rows: q['g'] = W.peak_g(q['x'])
    centre = defaultdict(list)
    for q in rows: centre[(q['hand'], q['drum'])].append(rel_yp(q['x']))
    centre = {k: np.median(np.array(v), 0) for k, v in centre.items()}
    ms = [W.train(rows, 'W0', s) for s in range(2)]
    _, cfg = core.load_components()
    sets = {k: (S.C['tests'][k]['truth'], S.C['tests'][k]['raw']) for k in ['seventh_01', 'seventh_12', 'seventh_15', 'pretest:round4', 'pretest:round5', 'pretest:round6']}
    s14 = S.C['sessions']['seventh_14']
    sets['song14'] = ([dict(t=e['csv_t'], drum=e['name']) for e in s14['events'] if e.get('name') and not e.get('unmapped') and not e.get('out_of_scope')], s14['raw'])
    res = {}
    for name, (truth, raw) in sets.items():
        hands = rt.world.read_raw(raw); prep = rt.world.base.legacy.prepare(hands)
        peaks, _ = detector.detect(hands, cfg); peaks = sorted(peaks, key=lambda p: p['emitted_ms'])
        out = {}
        for mode in ('固定歸零', '線上修正'):
            gap = {h: deque(maxlen=N) for h in 'LR'}; preds = []; shifts = []
            for p in peaks:
                h = p['hand']; s = prep[h]['rows'][p['available_index']]
                zy, zp = (s['absolute_yaw']-s['yaw']+180) % 360-180, s['absolute_pitch']-s['pitch']
                dy = dp = 0.0
                if mode == '線上修正' and len(gap[h]) >= 5:
                    g = np.median(np.array(gap[h]), 0); dy, dp = np.clip(g, -CLAMP, CLAMP)
                x = rt.feature(prep, p, zy-dy, zp-dp)
                if x is None: continue
                cls = ms[0][h][0]; q = np.mean([S.G.proba(m[h][1], x[None, :]) for m in ms], 0)[0]
                d = cls[int(q.argmax())]; preds.append(dict(t=p['peak_ms'], drum=d, score=float(q.max())))
                if q.max() >= CONF:
                    y0, p0 = rel_yp(rt.feature(prep, p, zy, zp)); c = centre[(h, d)]
                    gy = (c[0]-y0+180) % 360-180; gap[h].append((gy, c[1]-p0))
                shifts.append((round(float(dy), 1), round(float(dp), 1)))
            tt = np.array([g['t'] for g in truth]); lo, hi = tt.min()-200, tt.max()+200
            sc = S.score(truth, [q for q in preds if lo <= q['t'] <= hi], 0.0)
            out[mode] = dict(rate=sc['rate'], correct=sc['correct'], total=sc['total'], last_shift=shifts[-1] if shifts else None)
        res[name] = out; print(name, out, flush=True)
    (Path(__file__).parent/'out/online_zero_experiment.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding='utf-8')


if __name__ == '__main__':
    main()
