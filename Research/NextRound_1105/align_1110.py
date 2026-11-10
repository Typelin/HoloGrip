"""Which 11/10 pretest MIDI belongs to which pretest recording: MIDI-to-glove alignment rate for every pair."""
import sys
from pathlib import Path
KEYS = sys.argv[1].split(',')
import rounds_data as RD
L5 = RD.L5
PK = RD.ROOT/'11-10的包'
raws = {p.parent.name: p for p in sorted((PK/'Data/模型實測').glob('*/*_raw_100hz_*.csv')) if any(k in p.parent.name for k in KEYS)}
mids = sorted((PK/'今天').glob('測試*.mid'))
print('raws', list(raws), 'mids', [m.name for m in mids], flush=True)
if __name__ == '__main__':
    cand = {n: L5.hit_cand.work((n, str(p), 1.5, {}))[1] for n, p in raws.items()}
    for m in mids:
        for n in raws:
            _, info = RD.truth_of(m, cand[n])
            print(m.name, n[-30:], info, flush=True)
