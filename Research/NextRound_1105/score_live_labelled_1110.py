"""11/10 songs: live judgement (Raw/即時判斷) scored against the auto-labelled notes (hand + drum from
autolabel_day.py, same labels as training). Excluded / review notes are not scored. Each labelled note is
matched to a live prediction of the SAME glove (hand) within 80 ms (one-to-one, nearest in time)."""
import csv, json, sys
from collections import Counter
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
HERE = Path(__file__).resolve().parent; PK = HERE.parents[1]/'11-10的包'
LAB = HERE/'out/labels_R11'
RAWS = {'歌曲1改': 'S20261020_P01_song01_take01_raw_100hz_20261110_120700', '歌曲2改': 'S20261020_P01_song02_take01_raw_100hz_20261110_123116'}
res = {}
for name, stem in RAWS.items():
    d = json.loads(next(LAB.glob(f'{name}_自動標記_*.json')).read_text(encoding='utf-8'))
    truth = [dict(t=float(e['csv_ms']), hand=d['labels'][str(e['id'])]['mark'], drum=d['labels'][str(e['id'])]['drum'])
             for e in d['events'] if d['labels'][str(e['id'])]['mark'] in ('L', 'R')]
    P = [dict(hand=r['hand'], t=float(r['peak_ms']), drum=r['drum']) for r in csv.DictReader(open(PK/'Data/Raw/即時判斷'/f'{stem}_即時判斷.csv', encoding='utf-8-sig')) if r['accepted'] in ('True', '1')]
    ok = n = 0; cells = {}; conf = Counter(); miss = 0
    for h in 'LR':
        T = [g for g in truth if g['hand'] == h]; Q = [p for p in P if p['hand'] == h]
        dt = np.abs(np.array([g['t'] for g in T])[:, None]-np.array([p['t'] for p in Q])[None, :])
        cost = np.hstack([np.where(dt <= 80, dt, 1e9), np.full((len(T), len(T)), 1e6)]); r, c = linear_sum_assignment(cost)
        pair = {i: j for i, j in zip(r, c) if j < len(Q) and cost[i, j] <= 80}
        for i, g in enumerate(T):
            good = i in pair and Q[pair[i]]['drum'] == g['drum']; ok += good; n += 1
            k = f"{h}|{g['drum']}"; cells.setdefault(k, [0, 0]); cells[k][0] += good; cells[k][1] += 1
            if not good:
                w = 'miss' if i not in pair else Q[pair[i]]['drum']; conf[(k, w)] += 1; miss += i not in pair
    res[name] = dict(ok=ok, n=n, rate=round(ok/n, 4), missed=miss, cells=cells, errors=[[a, b, c] for (a, b), c in conf.most_common()])
    print(name, ok, '/', n, f'{100*ok/n:.1f}%', 'missed', miss)
    for k, v in sorted(cells.items()): print('   ', k, f'{v[0]}/{v[1]}', '' if v[0] == v[1] else {b: c for (a, b), c in conf.items() if a == k})
(HERE/'out/score_1110_live_songs_labelled.json').write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
