"""Per-day drum layout (snare-relative yaw/pitch at the hit, median) per hand, from training rows + pretests.
Shows whether days differ by a shift only or also by spacing (scale)."""
import sys, json
from pathlib import Path
import numpy as np, joblib
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import lodo5 as L5
K = L5.K
rows = joblib.load(K.OUT/'dataset2.joblib')['rows']+joblib.load(L5.OUT/'r8_cache.joblib')['rows']
DR = ['小鼓', '高音 Tom', '中音 Tom', '落地 Tom', 'Hi-Hat', 'Crash', 'Ride']
def yp(X): return L5.LF.yp(np.asarray(X))
res = {}
for h in 'LR':
    print('==', h)
    for D in L5.DAYS5:
        # per recording: snare-relative, then median over the day's recordings
        recs = {}
        for r in rows:
            if r['hand'] == h and L5.day_of(r['session']) == D: recs.setdefault(r['session'], []).append(r)
        rel = {d: [] for d in DR}
        for s, rs in recs.items():
            sn = [r['x'] for r in rs if r['drum'] == '小鼓']
            if len(sn) < 5: continue
            c = np.median(yp(sn), 0)
            for d in DR:
                v = [r['x'] for r in rs if r['drum'] == d]
                if len(v) >= 5: rel[d].append(np.median(yp(v), 0)-c)
        line = {d: np.median(v, 0).round(1).tolist() for d, v in rel.items() if v}
        res[f'{h}|{D}'] = line
        print(D, '  '.join(f"{d[:2]}({v[0]:+.0f},{v[1]:+.0f})" for d, v in line.items()))
(HERE/'out/layout_geom_1027.json').write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
