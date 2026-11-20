"""Score the live judgement of one 11/17 recording against a MIDI file (time-scale search).
Raw recordings use Data/Raw/即時判斷/<raw>_即時判斷.csv; pretests use Data/模型實測/<session>/predictions.csv.
Usage: python score_live_1117.py <raw time key, e.g. 104001> <midi path>"""
import csv, sys
from pathlib import Path
import numpy as np
ARGS = sys.argv[1:3]
import rounds_data as RD
import autolabel_1027 as AL
import verify_package_1027 as VP
L5 = RD.L5
PK = RD.ROOT/'11-17的包'


def find(key):
    raws = list((PK/'Data/Raw').glob(f'*_{key}.csv'))
    if raws: return raws[0], raws[0].parent/'即時判斷'/f'{raws[0].stem}_即時判斷.csv'
    raw = list((PK/'Data/模型實測').glob(f'*/*_{key}.csv'))[0]
    return raw, raw.parent/'predictions.csv'


if __name__ == '__main__':
    key, mid = ARGS[0], Path(ARGS[1])
    raw, live = find(key)
    P = [dict(hand=r['hand'], t=float(r['peak_ms']), peak_ms=float(r['peak_ms']), drum=r['drum']) for r in csv.DictReader(open(live, encoding='utf-8-sig')) if r['accepted'] in ('True', '1')]
    cand = L5.hit_cand.work((raw.stem, str(raw), 1.5, {}))[1]; ct = np.array(sorted(x['t'] for x in cand))
    ev = [e for e in AL.legacy.midi(mid)['events'] if e['vel'] >= AL.GHOST_VEL and e['name'] in RD.K.DRUMS]
    mt = np.array([e['t'] for e in ev])
    s, off, rate = max(((s,)+tuple(AL.align(mt*s, ct)) for s in np.r_[1.0, np.arange(0.88, 1.16, 0.0005)]), key=lambda x: x[2])
    truth = []
    for e in ev:
        t = e['t']*s+off; near = [x for x in cand if abs(x['t']-t) <= 60]
        truth.append(dict(id=e['id'], t=t, drum=e['name'], hand=max(near, key=lambda x: x['g'])['hand'] if near else None))
    ok, n, cells = VP.score(truth, P)
    print(f'== {raw.name}  {mid.name}  scale {s:.4f} offset {off:.0f} align {rate:.3f}  midi {len(ev)}  live {len(P)}  score {ok}/{n} = {ok/n:.4f}', flush=True)
    for k, v in sorted(cells.items()): print(f'     {k:12s} {v[0]:3d}/{v[1]:3d}', v[2] if v[0] < v[1] else '')
