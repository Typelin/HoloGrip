"""11/17 (人員3) per-take hit pose: zero-relative yaw/pitch at each hit from the field engine (11-24的包, model h5),
relative to the take's own snare median, with spread (IQR) and hit rate. Answer key from each take's MIDI
(time-scale search; hand = strongest candidate within 60 ms; drills are single-hand blocks). Read-only."""
import csv, sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]; PK = ROOT/'11-24的包'; PK17 = ROOT/'11-17的包'
sys.path[:0] = [str(PK/'Program'), str(HERE)]
M = PK17/'今天'/'雲端_11-17'
TAKES = [('pre0952', 'pre:095244', '1117-1 LR 20.mid'), ('song', '104001', '1117-2.1 All  (fixed).mid'),
         ('L_HRG', '111012', '1117-3 L5 T1RidFT  (fixed).mid'), ('R_HRG', '111810', '1117-3 R5 T1RidFT  (fixed).mid')]


def raw_of(key):
    if key.startswith('pre:'):
        return next(p for p in (PK17/'Data/模型實測').glob(f'*_{key[4:]}_*/*_raw_100hz_*.csv'))
    return next((PK17/'Data/Raw').glob(f'*_{key}.csv'))


def truth_of(raw, mid):
    import rounds_data as RD, autolabel_1027 as AL
    cand = RD.L5.hit_cand.work((raw.stem, str(raw), 1.5, {}))[1]; ct = np.array(sorted(x['t'] for x in cand))
    ev = [e for e in AL.legacy.midi(mid)['events'] if e['vel'] >= AL.GHOST_VEL and e['name'] in RD.K.DRUMS]
    mt = np.array([e['t'] for e in ev])
    s, off, _ = max(((s,)+tuple(AL.align(mt*s, ct)) for s in np.r_[1.0, np.arange(0.95, 1.10, 0.0005)]), key=lambda x: x[2])
    out = []
    for e in ev:
        t = e['t']*s+off; near = [x for x in cand if abs(x['t']-t) <= 60]
        if near: out.append(dict(t=t, drum=e['name'], hand=max(near, key=lambda x: x['g'])['hand']))
    return out


def work(job):
    name, key, mid = job
    import seventh_core as core
    raw = raw_of(key); truth = truth_of(raw, M/mid)
    b, config = core.load_components()
    hands = core.trajectory.read_raw(str(raw))
    with open(raw, encoding='utf-8-sig', newline='') as f: source = list(csv.DictReader(f))
    engine = core.Engine(b, config, snare_anchor=5); rt = engine.runtime; rec = []; orig = rt._classify
    def spy(hand, x):
        rec.append(core.runtime._yaw_pitch(x).tolist()); return orig(hand, x)
    rt._classify = spy; out = []
    for _, h, s in sorted((s['row_index'], h, s) for h, a in hands.items() for s in a):
        row = source[s['row_index']]; n0 = len(rec)
        p, _ = engine.push(h, s, dict(ax=s['ax'], ay=s['ay'], az=s['az'], yaw=float(row['yaw_deg']), pitch=float(row['pitch_deg']), roll=s['roll']))
        for q, yp in zip(p, rec[n0:]): out.append((q['hand'], q['peak_ms'], yp[0], yp[1], q['drum']))
    cell = defaultdict(list); gaps = defaultdict(list)
    for g in truth:
        near = [o for o in out if o[0] == g['hand'] and abs(o[1]-g['t']) <= 60]
        if near: cell[(g['hand'], g['drum'])].append(min(near, key=lambda o: abs(o[1]-g['t'])))
    for h in 'LR':
        ts = sorted(g['t'] for g in truth if g['hand'] == h and g['drum'] != '小鼓')
        gaps[h] = list(np.diff(ts)) if len(ts) > 2 else []
    return name, dict(cell), {h: (float(np.median(v)) if v else None) for h, v in gaps.items()}


if __name__ == '__main__':
    order = ['小鼓', 'Hi-Hat', 'Crash', '高音 Tom', '中音 Tom', 'Ride', '落地 Tom']
    with Pool(5) as pool: R = {n: (c, g) for n, c, g in pool.imap(work, TAKES)}
    for h in 'LR':
        print(f'===== hand {h}: yaw,pitch vs own snare | IQR yaw | n | correct%   (median gap between non-snare hits)')
        for name, (cell, gap) in R.items():
            if (h, '小鼓') not in cell: continue
            sn = np.array([(o[2], o[3]) for o in cell[(h, '小鼓')]]); ref = np.array([np.degrees(np.angle(np.mean(np.exp(1j*np.radians(sn[:, 0]))))), np.median(sn[:, 1])]); parts = []
            for d in order[1:]:
                v = cell.get((h, d))
                if not v or len(v) < 5: continue
                a = np.array([(o[2], o[3]) for o in v]); acc = np.mean([o[4] == d for o in v])
                w = (a[:, 0]-ref[0]+180) % 360-180; dy = np.median(w); iqr = np.subtract(*np.percentile(w, [75, 25]))
                t = np.array([o[1] for o in v])/60000.0
                slope = np.polyfit(t-t.mean(), w, 1)[0] if np.ptp(t) > 0.5 else 0.0
                ab = (a[:, 0]+180) % 360-180
                parts.append(f"{d[:2]} abs{np.median(ab):+5.0f} {dy:+4.0f},{np.median(a[:, 1])-ref[1]:+4.0f} ±{iqr:3.0f} drift{slope:+4.0f}/min n{len(v)} {100*acc:3.0f}%")
            parts.insert(0, f"snare abs {ref[0]:+5.0f},{ref[1]:+4.0f}")
            if parts: print(f"  {name:8s} gap {gap[h] or 0:5.0f}ms  " + ' | '.join(parts))
