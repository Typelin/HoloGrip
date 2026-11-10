"""Air-drum vs real-drum on 11/10: replay each take through the field engine and record, for every accepted hit,
the zero-relative yaw/pitch at the hit (before any correction), peak g, the engine's drum and score.
Blocks = pauses per hand (> 2.2 x median gap or 1.2 s). Read-only.  -> out/pose_air_1110.json"""
from __future__ import annotations
import csv, json, sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import joblib
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]; CODE = '11-10的包'
MODEL = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1].endswith('.joblib') else 'out/model_r10c.joblib'
sys.path[:0] = [str(ROOT/CODE/'Program'), str(HERE)]
import seventh_core as core
MD = ROOT/CODE/'Data/模型實測'
TAKES = {'115737': '人員1 真鼓 正打', '124408': '人員1 空氣', '133616': '人員3 真鼓 正打', '150827': '人員3 真鼓 正打(MIDI)', '134421': '人員3 真鼓 反打',
         '161301': '人員3 空氣 正打', '161732': '人員3 空氣 反打'}


def work(key):
    raw = next(next(p for p in MD.glob('*') if key in p.name).glob('*_raw_100hz_*.csv'))
    b = joblib.load(HERE/MODEL); core.runtime.attach_boosters(b); config = core.load_components()[1]
    hands = core.trajectory.read_raw(raw)
    with open(raw, encoding='utf-8-sig', newline='') as f: source = list(csv.DictReader(f))
    engine = core.Engine(b, config, snare_anchor=5); rt = engine.runtime; rec = []
    orig = rt._classify
    def spy(hand, x):
        rec.append((hand, core.runtime._yaw_pitch(x).tolist())); return orig(hand, x)
    rt._classify = spy; out = []
    for _, h, s in sorted((s['row_index'], h, s) for h, a in hands.items() for s in a):
        row = source[s['row_index']]; n0 = len(rec)
        p, _ = engine.push(h, s, dict(ax=s['ax'], ay=s['ay'], az=s['az'], yaw=float(row['yaw_deg']), pitch=float(row['pitch_deg']), roll=s['roll']))
        for q, (hh, yp) in zip(p, rec[n0:]):
            out.append(dict(hand=q['hand'], t=q['peak_ms'], g=q['peak_g'], drum=q['drum'], score=q['model_score'], yaw=yp[0], pitch=yp[1]))
    return key, out, dict(rejected=dict(rt.rejected), anchored={h: rt.anchored[h] for h in 'LR'}, zero_rot=dict(rt.zero_rot))


if __name__ == '__main__':
    res = {}
    with Pool(4) as pool:
        for key, hits, info in pool.imap(work, list(TAKES)):
            res[key] = dict(name=TAKES[key], info=info, hits=hits)
            print('=====', key, TAKES[key], 'hits', len(hits), info)
            for h in 'LR':
                Q = sorted((x for x in hits if x['hand'] == h), key=lambda x: x['t']); t = np.array([x['t'] for x in Q])/1000
                if len(Q) < 3: continue
                dt = np.diff(t); cuts = [i+1 for i in np.where(dt > max(2.2*np.median(dt), 1.2))[0]]
                for seg in np.split(np.arange(len(Q)), cuts):
                    if len(seg) < 3: continue
                    S = [Q[i] for i in seg]; c = Counter(x['drum'] for x in S).most_common(3)
                    print(f"  {h} {t[seg[0]]:6.1f}s n={len(seg):3d} yaw {np.median([x['yaw'] for x in S]):7.1f} pitch {np.median([x['pitch'] for x in S]):6.1f}"
                          f"  spread {np.std([x['yaw'] for x in S]):4.1f}/{np.std([x['pitch'] for x in S]):4.1f}  g {np.median([x['g'] for x in S]):4.1f}  score {np.median([x['score'] for x in S]):.2f}  {c}")
    (HERE/'out/pose_air_1110.json').write_text(json.dumps(res, ensure_ascii=False), encoding='utf-8')
