"""Score one pretest recording (live predictions.csv) against a given MIDI file. Usage: python score_one_pretest_1110.py <take dir key> <midi path>"""
import csv, sys, glob
from collections import Counter
from pathlib import Path
KEY, MID = sys.argv[1], Path(sys.argv[2])
import rounds_data as RD
import verify_package_1027 as VP
L5 = RD.L5
PK = RD.ROOT/'11-10的包'
if __name__ == '__main__':
    D = [p for p in (PK/'Data/模型實測').glob('*') if KEY in p.name][0]
    raw = list(D.glob('*_raw_100hz_*.csv'))[0]
    P = [dict(hand=r['hand'], t=float(r['peak_ms']), peak_ms=float(r['peak_ms']), drum=r['drum']) for r in csv.DictReader(open(D/'predictions.csv', encoding='utf-8-sig')) if r['accepted'] in ('True', '1')]
    cand = L5.hit_cand.work((raw.stem, str(raw), 1.5, {}))[1]
    truth, info = RD.truth_of(MID, cand)
    print(D.name, 'live hits', len(P), 'cand', len(cand), 'midi notes', len(truth), info, flush=True)
    print('midi drums', dict(Counter(g['drum'] for g in truth)))
    ok, n, cells = VP.score(truth, P)
    print('score', ok, '/', n, round(ok/n, 4))
    for k, v in sorted(cells.items()): print('   ', k, v[0], '/', v[1], v[2] if v[0] < v[1] else '')
