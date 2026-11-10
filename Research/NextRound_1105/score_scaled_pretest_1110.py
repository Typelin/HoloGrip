"""Score one pretest whose MIDI has a time-scale error (tempo mismatch on export).
Searches a global time scale, then builds the answer key like rounds_data.truth_of (hand = strongest candidate within 60 ms).
Usage: python score_scaled_pretest_1110.py <take dir key> <midi path>"""
import csv, sys
from collections import Counter
from pathlib import Path
import numpy as np
KEY, MID = sys.argv[1], Path(sys.argv[2])
import rounds_data as RD
import autolabel_1027 as AL
import verify_package_1027 as VP
L5 = RD.L5
PK = RD.ROOT/'11-10的包'
if __name__ == '__main__':
    D = [p for p in (PK/'Data/模型實測').glob('*') if KEY in p.name][0]
    raw = list(D.glob('*_raw_100hz_*.csv'))[0]
    P = [dict(hand=r['hand'], t=float(r['peak_ms']), peak_ms=float(r['peak_ms']), drum=r['drum']) for r in csv.DictReader(open(D/'predictions.csv', encoding='utf-8-sig')) if r['accepted'] in ('True', '1')]
    cand = L5.hit_cand.work((raw.stem, str(raw), 1.5, {}))[1]
    ct = np.array(sorted(x['t'] for x in cand))
    ev = [e for e in AL.legacy.midi(MID)['events'] if e['vel'] >= AL.GHOST_VEL and e['name'] in RD.K.DRUMS]
    mt = np.array([e['t'] for e in ev])
    best = None
    for s in np.arange(0.95, 1.15, 0.0005):
        off, rate = AL.align(mt*s, ct)
        if best is None or rate > best[2]: best = (s, off, rate)
    s, off, rate = best
    truth = []
    for e in ev:
        t = e['t']*s+off; near = [x for x in cand if abs(x['t']-t) <= 60]
        truth.append(dict(id=e['id'], t=t, drum=e['name'], hand=max(near, key=lambda x: x['g'])['hand'] if near else None))
    print(D.name, 'scale', round(s, 4), 'offset', round(off, 1), 'align_rate', round(rate, 4), 'hands', dict(Counter(g['hand'] for g in truth)), 'live hits', len(P))
    ok, n, cells = VP.score(truth, P)
    print('score', ok, '/', n, round(ok/n, 4))
    for k, v in sorted(cells.items()): print('   ', k, v[0], '/', v[1], v[2] if v[0] < v[1] else '')
    seq = [(g['hand'], g['drum']) for g in truth]
    blocks = [(h, d, sum(1 for _ in grp)) for (h, d), grp in __import__('itertools').groupby(seq)]
    print('truth blocks', ' '.join(f'{h}{d}×{c}' for h, d, c in blocks))
