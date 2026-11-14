"""Score the collector's live judgement (Data/Raw/即時判斷) of one 11/12 recording against a MIDI file (time-scale search).
Usage: python score_live_one_1112.py <raw time key, e.g. 113853> <midi path>"""
import csv, sys
from pathlib import Path
import numpy as np
ARGS = sys.argv[1:3]
import rounds_data as RD
import autolabel_1027 as AL
import verify_package_1027 as VP
L5 = RD.L5
PK = RD.ROOT/'11-12的包'
if __name__ == '__main__':
    key, mid = ARGS[0], Path(ARGS[1])
    raw = [p for p in (PK/'Data/Raw').glob(f'*_{key}.csv')][0]
    live = raw.parent/'即時判斷'/f'{raw.stem}_即時判斷.csv'
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
    print(f'== {raw.name}  {mid.name}  scale {s:.4f} align {rate:.3f}  midi {len(ev)}  live {len(P)}  score {ok}/{n} = {ok/n:.4f}', flush=True)
    for k, v in sorted(cells.items()): print(f'     {k:12s} {v[0]:3d}/{v[1]:3d}', v[2] if v[0] < v[1] else '')


def drum_only(truth, P, tol=60.0):
    """One-to-one matching ignoring the answer-key hand: each MIDI note may use one live prediction within tol ms,
    preferring a same-drum prediction. Also reports hand-checked score on notes with no other note within 40 ms."""
    order = sorted(range(len(P)), key=lambda j: P[j]['t']); used = set(); ok = 0; per = {}
    times = np.array(sorted(g['t'] for g in truth))
    solo_ok = solo_n = 0
    for g in sorted(truth, key=lambda g: g['t']):
        near = [j for j in order if j not in used and abs(P[j]['t']-g['t']) <= tol]
        same = [j for j in near if P[j]['drum'] == g['drum']]
        pick = min(same or near, key=lambda j: abs(P[j]['t']-g['t'])) if near else None
        good = pick is not None and P[pick]['drum'] == g['drum']
        if pick is not None: used.add(pick)
        ok += good
        c = per.setdefault(g['drum'], [0, 0, {}]); c[0] += good; c[1] += 1
        if not good:
            w = 'miss' if pick is None else P[pick]['drum']; c[2][w] = c[2].get(w, 0)+1
        solo = int(np.sum(np.abs(times-g['t']) <= 40)) == 1
        if solo and g['hand']:
            solo_n += 1; solo_ok += good and P[pick]['hand'] == g['hand']
    return ok, len(truth), per, solo_ok, solo_n


if __name__ == '__main__':
    ok2, n2, per, so, sn = drum_only(truth, P)
    print(f'-- drum only (one-to-one, hand ignored): {ok2}/{n2} = {ok2/n2:.4f}')
    for k, v in sorted(per.items(), key=lambda kv: -kv[1][1]): print(f'     {k:10s} {v[0]:3d}/{v[1]:3d} {v[0]/v[1]:.3f}', v[2] if v[0] < v[1] else '')
    print(f'-- single notes (no other note within 40 ms), hand+drum: {so}/{sn} = {so/max(sn,1):.4f}')
