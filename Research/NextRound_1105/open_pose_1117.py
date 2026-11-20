"""11/17: zero-relative yaw/pitch and top-3 drum scores of the first hits of each 七鼓 recording (11-24的包 engine, model h5).
Used to see why the opening snares of the late drum-kit recordings were judged Hi-Hat. Read-only."""
import csv, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]; PK = ROOT/'11-24的包'; PK17 = ROOT/'11-17的包'
sys.path[:0] = [str(PK/'Program'), str(HERE)]
import seventh_core as core
KEYS = sys.argv[1:] or ['095244', '114625', '114746', '114938']
b, config = core.load_components()
for key in KEYS:
    raw = next((PK17/'Data/模型實測').glob(f'*_{key}_*/*_raw_100hz_*.csv'))
    hands = core.trajectory.read_raw(str(raw))
    with open(raw, encoding='utf-8-sig', newline='') as f: source = list(csv.DictReader(f))
    engine = core.Engine(b, config, snare_anchor=5); rt = engine.runtime; rec = []; orig = rt._classify
    def spy(hand, x, rec=rec, orig=orig):
        out = orig(hand, x); rec.append((core.runtime._yaw_pitch(x).tolist(), out)); return out
    rt._classify = spy; got = {'L': [], 'R': []}
    for _, h, s in sorted((s['row_index'], h, s) for h, a in hands.items() for s in a):
        row = source[s['row_index']]; n0 = len(rec)
        p, _ = engine.push(h, s, dict(ax=s['ax'], ay=s['ay'], az=s['az'], yaw=float(row['yaw_deg']), pitch=float(row['pitch_deg']), roll=s['roll']))
        for q, (yp, out) in zip(p, rec[n0:]):
            if len(got[q['hand']]) < 12: got[q['hand']].append((q['peak_ms'], yp, q['drum'], out))
        if all(len(v) >= 12 for v in got.values()): break
    print('==', key)
    for h in 'LR':
        for t, yp, d, out in got[h]:
            sc = out if isinstance(out, dict) else {}
            pr = np.round(np.asarray(out[0]), 2).tolist() if isinstance(out, tuple) else out
            print(f"  {h} {t:7.0f} yaw {yp[0]:+6.1f} pitch {yp[1]:+5.1f} -> {d}  probs {pr}  classes {list(b['classes'])}" if t == got[h][0][0] else f"  {h} {t:7.0f} yaw {yp[0]:+6.1f} pitch {yp[1]:+5.1f} -> {d}  probs {pr}")
