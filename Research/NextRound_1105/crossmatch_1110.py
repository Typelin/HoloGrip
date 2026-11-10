"""Cross-match every 11/10 recording (模型實測 takes + Raw songs) against every MIDI under 今天/ (recursive), with a global time-scale search.
Prints the best MIDI per recording. Read-only."""
import hashlib, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import rounds_data as RD
import autolabel_1027 as AL
L5 = RD.L5
PK = RD.ROOT/'11-10的包'
SCALES = np.r_[1.0, np.arange(0.88, 1.16, 0.0025)]


def mids():
    out, seen = [], set()
    for m in sorted((PK/'今天').rglob('*.mid')):
        h = hashlib.sha256(m.read_bytes()).hexdigest()
        if h in seen: continue
        seen.add(h)
        ev = [e for e in AL.legacy.midi(m)['events'] if e['vel'] >= AL.GHOST_VEL and e['name'] in RD.K.DRUMS]
        out.append((m.name, np.array([e['t'] for e in ev])))
    return out


def work(raw):
    cand = L5.hit_cand.work((raw.stem, str(raw), 1.5, {}))[1]
    if len(cand) < 30: return raw, len(cand), []
    ct = np.array(sorted(x['t'] for x in cand)); res = []
    for name, mt in MIDS:
        best = max(((s,)+tuple(AL.align(mt*s, ct)) for s in SCALES), key=lambda x: x[2])
        res.append((round(best[2], 3), round(best[0], 4), name, len(mt)))
    return raw, len(cand), sorted(res, reverse=True)[:3]


MIDS = mids()
if __name__ == '__main__':
    raws = sorted((PK/'Data/模型實測').glob('*/*_raw_100hz_*.csv'))+sorted((PK/'Data/Raw').glob('*_raw_100hz_*.csv'))
    print('mids', [n for n, _ in MIDS], flush=True)
    with Pool(6) as pool:
        for raw, n, top in pool.imap(work, raws):
            print(f'{raw.stem[-40:]:42s} cand {n:4d}', ' | '.join(f'{r} x{s} {nm} ({k})' for r, s, nm, k in top), flush=True)
