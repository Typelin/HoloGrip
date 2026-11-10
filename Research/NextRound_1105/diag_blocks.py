"""Provisional pretest score before the MIDI arrives: each hand's hits are split into blocks by pauses > 3 s
and labelled with the fixed pretest order (snare 10 + 20, Hi-Hat, Crash, high, mid, Ride, floor x 20)."""
import csv, sys
RAW = sys.argv[1]   # read before imports that rewrite sys.argv
from collections import Counter
from pathlib import Path
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT/'第十次的包_暫定'/'Program'), str(HERE)]
import seventh_core as core
ORDER = ['小鼓', '小鼓', 'Hi-Hat', 'Crash', '高音 Tom', '中音 Tom', 'Ride', '落地 Tom']
def blocks(P):
    out = {}
    for h in 'LR':
        a = sorted([p for p in P if p['hand'] == h], key=lambda p: float(p['peak_ms'])); b = 0; lab = []
        for i, p in enumerate(a):
            if i and float(p['peak_ms'])-float(a[i-1]['peak_ms']) > 3000: b += 1
            lab.append((p, ORDER[min(b, 7)]))
        out[h] = (lab, b+1)
    return out
def rep(name, P):
    B = blocks(P); tot = ok = 0; lines = []
    for h in 'LR':
        lab, nb = B[h]; c = {}
        for p, d in lab: c.setdefault(d, Counter())[p['drum']] += 1
        for d in dict.fromkeys(ORDER):
            k = c.get(d, Counter()); n = sum(k.values()); o = k[d]; ok += o; tot += n
            lines.append(f"    {h}|{d} {o}/{n} " + ('' if o == n else str({x: v for x, v in k.items() if x != d})))
        lines.append(f'    {h} blocks {nb}')
    print(f'{name}: {ok}/{tot} = {100*ok/tot:.1f}%'); print('\n'.join(lines))
pred = list(csv.DictReader(open(Path(RAW).parent/'predictions.csv', encoding='utf-8-sig')))
rep('field predictions.csv', pred)
bundle, config = core.load_components(); rt_mod = core.runtime
hands = core.trajectory.read_raw(RAW); src = list(csv.DictReader(open(RAW, encoding='utf-8-sig', newline='')))
def stream(anchor=rt_mod.PRETEST_SNARE_ANCHOR, force_snare=False, b=bundle):
    saved = dict(rt_mod.ANCHOR_RULE)
    if force_snare: rt_mod.ANCHOR_RULE.update(snare_in_top=99, max_pose_deg=1e9)
    try:
        eng = core.Engine(b, config, snare_anchor=anchor); out = []
        for _, h, s in sorted((s['row_index'], h, s) for h, a in hands.items() for s in a):
            row = src[s['row_index']]
            p, _ = eng.push(h, s, dict(ax=s['ax'], ay=s['ay'], az=s['az'], yaw=float(row['yaw_deg']), pitch=float(row['pitch_deg']), roll=s['roll']))
            out.extend(p)
        print('  anchored', eng.runtime.anchored, 'zero_rot', eng.runtime.zero_rot)
    finally: rt_mod.ANCHOR_RULE.clear(); rt_mod.ANCHOR_RULE.update(saved)
    return out
rep('replay field settings', stream())
rep('replay: first 5 hits forced as snare anchor', stream(force_snare=True))
rep('replay: first 10 hits forced as snare anchor', stream(anchor=10, force_snare=True))
