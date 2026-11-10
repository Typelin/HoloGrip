"""Provisional check of a pretest before its MIDI arrives: align yesterday's pretest answer key (same score)
to today's detected hits by a single time offset, then score the field predictions and engine variants."""
import csv, json, sys
RAW = sys.argv[1]   # read before imports that rewrite sys.argv
from collections import Counter
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT/'第十次的包_暫定'/'Program'), str(HERE)]
import seventh_core as core
import rounds_data as RD
import verify_package_1027 as VP
import truth_clean_1027 as TC

RD.setup(); L5, H = RD.L5, RD.H
ref = 'pretest:R9'; truth, _ = TC.dedupe(L5.TRUTH[ref], H.CAND[ref])
truth = sorted(truth, key=lambda g: g['t'])
for h in 'LR':
    seq = [g['drum'] for g in truth if g['hand'] == h]; runs = []
    for d in seq:
        if runs and runs[-1][0] == d: runs[-1][1] += 1
        else: runs.append([d, 1])
    print('answer order', h, runs)
pred = [dict(hand=r['hand'], peak_ms=float(r['peak_ms']), drum=r['drum']) for r in csv.DictReader(open(Path(RAW).parent/'predictions.csv', encoding='utf-8-sig'))]
tt = np.array([g['t'] for g in truth]); pt = np.array([p['peak_ms'] for p in pred])
best = max(((sum(np.min(np.abs(pt[:, None]-(tt[None, :]+o)), 1) <= 80), o) for o in np.arange(pt.min()-tt.min()-20000, pt.min()-tt.min()+20000, 10)))
print('offset', best[1], 'hits within 80 ms', best[0], 'of', len(pt))
sh = [dict(g, t=g['t']+best[1]) for g in truth]
def rep(name, P):
    ok, n, cells = VP.score(sh, P)
    print(f'{name}: {ok}/{n} = {100*ok/n:.1f}%')
    for k, v in sorted(cells.items()): print('   ', k, f'{v[0]}/{v[1]}', v[2] if v[0] < v[1] else '')
rep('field predictions.csv', pred)
bundle, config = core.load_components()
hands = core.trajectory.read_raw(RAW)
src = list(csv.DictReader(open(RAW, encoding='utf-8-sig', newline='')))
def stream(b, anchor=core.runtime.PRETEST_SNARE_ANCHOR, rule=None):
    old = dict(core.runtime.ANCHOR_RULE) if hasattr(core.runtime, 'ANCHOR_RULE') else None
    eng = core.Engine(b, config, snare_anchor=anchor); out = []
    for _, h, s in sorted((s['row_index'], h, s) for h, a in hands.items() for s in a):
        row = src[s['row_index']]
        p, _ = eng.push(h, s, dict(ax=s['ax'], ay=s['ay'], az=s['az'], yaw=float(row['yaw_deg']), pitch=float(row['pitch_deg']), roll=s['roll']))
        out.extend(p)
    rt = eng.runtime
    print('  anchored', rt.anchored, 'zero_rot', rt.zero_rot)
    return sorted(out, key=lambda p: (p['peak_ms'], p['hand']))
rep('replay (field settings)', stream(bundle))
json.dump(dict(offset=float(best[1])), open(RD.OUT/'diag_today.json', 'w'))
