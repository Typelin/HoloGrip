"""Score the collector's live judgement (Raw/即時判斷) of afternoon 11/10 recordings against their MIDI (time-scale search).
Answer key as rounds_data.truth_of: MIDI notes aligned to glove candidates, hand = strongest candidate within 60 ms."""
import csv, json
from collections import Counter
import numpy as np
import rounds_data as RD
import autolabel_1027 as AL
import verify_package_1027 as VP
L5 = RD.L5
PK = RD.ROOT/'11-10的包'; C2 = PK/'今天'/'雲端_11-10-2'
PAIRS = [('155247', '1110 LHM fixed.mid'), ('155804', '1110 LMR fixed.mid'), ('160236', '1110 LRG fixed.mid'),
         ('153748', '1101 測試3 fixed.mid'), ('154120', 'STEM 計畫-電子鼓數據搜集(1110)-歌曲2 fixed.als.mid'),
         ('135707', 'STEM 計畫-電子鼓數據搜集(1110)-歌曲1 fixed.als.mid')]
if __name__ == '__main__':
    out = {}
    for key, mid in PAIRS:
        raw = [p for p in (PK/'Data/Raw').glob(f'*_{key}.csv')][0]
        live = raw.parent/'即時判斷'/f'{raw.stem}_即時判斷.csv'
        P = [dict(hand=r['hand'], t=float(r['peak_ms']), peak_ms=float(r['peak_ms']), drum=r['drum']) for r in csv.DictReader(open(live, encoding='utf-8-sig')) if r['accepted'] in ('True', '1')]
        cand = L5.hit_cand.work((raw.stem, str(raw), 1.5, {}))[1]; ct = np.array(sorted(x['t'] for x in cand))
        ev = [e for e in AL.legacy.midi(C2/mid)['events'] if e['vel'] >= AL.GHOST_VEL and e['name'] in RD.K.DRUMS]
        mt = np.array([e['t'] for e in ev])
        s, off, rate = max(((s,)+tuple(AL.align(mt*s, ct)) for s in np.r_[1.0, np.arange(0.88, 1.16, 0.0005)]), key=lambda x: x[2])
        truth = []
        for e in ev:
            t = e['t']*s+off; near = [x for x in cand if abs(x['t']-t) <= 60]
            truth.append(dict(id=e['id'], t=t, drum=e['name'], hand=max(near, key=lambda x: x['g'])['hand'] if near else None))
        ok, n, cells = VP.score(truth, P)
        out[raw.stem] = dict(midi=mid, scale=round(s, 4), align=round(rate, 4), live=len(P), ok=ok, n=n, cells={k: [v[0], v[1], v[2]] for k, v in cells.items()})
        print(f'== {raw.stem[-38:]}  {mid[-26:]}  x{s:.4f} align {rate:.3f}  live {len(P)}  score {ok}/{n} = {ok/n:.3f}', flush=True)
        for k, v in sorted(cells.items()):
            if v[0] < v[1] or v[1] >= 5: print(f'     {k:12s} {v[0]:3d}/{v[1]:3d}', v[2] if v[0] < v[1] else '')
    (RD.OUT/'score_1110_afternoon_live.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding='utf-8')
