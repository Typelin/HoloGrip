"""人員3 (R12) left/right hand pose per drum in every take with an answer key (pretests + labelled takes):
zero-relative yaw/pitch at the hit (uncorrected), relative to that take's own snare median. Read-only."""
import csv, json, sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path
import numpy as np, joblib
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]; CODE = '11-10的包'
sys.path[:0] = [str(ROOT/CODE/'Program'), str(HERE)]
import seventh_core as core
DAY = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1].startswith('R') else 'R12'


def jobs():
    import rounds_data as RD
    RD.setup(); c = RD.config()[DAY]; J = []
    for nm, raw in RD.pretest_names(DAY, c).items(): J.append((nm, str(raw), [dict(t=g['t'], hand=g['hand'], drum=g['drum']) for g in RD.L5.TRUTH[nm] if g['hand']]))
    for nm, d in RD.docs_of(DAY, c).items():
        tr = [dict(t=float(e['csv_ms']), hand=d['labels'][str(e['id'])]['mark'], drum=d['labels'][str(e['id'])]['drum']) for e in d['events'] if d['labels'][str(e['id'])]['mark'] in ('L', 'R')]
        J.append((nm, d['raw_csv'], tr))
    return J


def work(job):
    nm, raw, truth = job
    b = joblib.load(HERE/'out/model_r10c.joblib'); core.runtime.attach_boosters(b); config = core.load_components()[1]
    hands = core.trajectory.read_raw(raw)
    with open(raw, encoding='utf-8-sig', newline='') as f: source = list(csv.DictReader(f))
    engine = core.Engine(b, config, snare_anchor=5); rt = engine.runtime; rec = []; orig = rt._classify
    def spy(hand, x):
        rec.append(core.runtime._yaw_pitch(x).tolist()); return orig(hand, x)
    rt._classify = spy; out = []
    for _, h, s in sorted((s['row_index'], h, s) for h, a in hands.items() for s in a):
        row = source[s['row_index']]; n0 = len(rec)
        p, _ = engine.push(h, s, dict(ax=s['ax'], ay=s['ay'], az=s['az'], yaw=float(row['yaw_deg']), pitch=float(row['pitch_deg']), roll=s['roll']))
        for q, yp in zip(p, rec[n0:]): out.append((q['hand'], q['peak_ms'], yp[0], yp[1], q['drum']))
    cell = defaultdict(list)
    for g in truth:
        near = [o for o in out if o[0] == g['hand'] and abs(o[1]-g['t']) <= 60]
        if near: cell[(g['hand'], g['drum'])].append(min(near, key=lambda o: abs(o[1]-g['t'])))
    return nm, {k: v for k, v in cell.items()}


if __name__ == '__main__':
    J = jobs(); order = ['小鼓', 'Hi-Hat', 'Crash', '高音 Tom', '中音 Tom', 'Ride', '落地 Tom']
    with Pool(6) as pool: R = dict(pool.imap(work, J))
    for h in 'LR':
        print(f'===== {DAY} hand {h}: yaw,pitch relative to the take\'s own snare (n, correct%)')
        for nm, cell in R.items():
            if (h, '小鼓') not in cell: ref = None
            else: ref = np.median(np.array([(o[2], o[3]) for o in cell[(h, '小鼓')]]), 0)
            parts = []
            for d in order:
                v = cell.get((h, d))
                if not v or len(v) < 5: continue
                m = np.median(np.array([(o[2], o[3]) for o in v]), 0); acc = np.mean([o[4] == d for o in v])
                dy = ((m[0]-(ref[0] if ref is not None else 0))+180) % 360-180; dp = m[1]-(ref[1] if ref is not None else 0)
                parts.append(f"{d[:2]} {dy:+5.0f},{dp:+4.0f} ({len(v)},{100*acc:.0f}%)")
            print(f"  {nm[-14:]:14s}{'' if ref is not None else '[abs]'} " + ' | '.join(parts))
