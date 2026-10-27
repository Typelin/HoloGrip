"""Why did the 10/27 pretest left hand fail on 中音 Tom / Ride / 落地 Tom?

Compares the zero-relative stick pose at the hit (x[95:98], rotvec at dt=0) per left-hand drum for
training rounds R4-R7, the 10/27 pretest (truth from MIDI, hand from glove) and today's single-hand takes.
Diagnosis only; nothing is trained here.
"""
from __future__ import annotations
import json, sys, warnings
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
warnings.filterwarnings('ignore')
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import autolabel_1027 as L
import joblib

DR = ['小鼓', '高音 Tom', '中音 Tom', '落地 Tom', 'Hi-Hat', 'Crash', 'Ride']
PRE_RAW = next((L.P8/'Data/模型實測').glob('S20261027*/S20261027_*_raw_100hz_*.csv'))


def ang(x):
    """yaw / pitch (deg) of the hit pose relative to drumhead zero."""
    e = Rotation.from_rotvec(np.asarray(x)[95:98]).as_euler('ZYX', degrees=True)
    return e[0], e[1]


def match(ev, items, hand=None):
    out = []
    used = set()
    for e in ev:
        best = None
        for j, c in enumerate(items):
            p = c['peak']
            if j in used or (hand and p['hand'] != hand): continue
            d = abs(p['peak_ms']-e['csv_t'])
            if d <= 60 and (best is None or d < best[0]): best = (d, j)
        if best: used.add(best[1]); out.append((e, items[best[1]]))
    return out


def main():
    C = joblib.load(L.P7/'Data/實驗_姿態模型自動標記_20261020/實驗快取.joblib'); L.DET = C['detector']
    stats = defaultdict(list)
    for r in joblib.load(L.ROOT/'Research/BestModel_20261021/out/dataset.joblib')['rows']:
        if r['hand'] == 'L': stats[(r['group'][:2], r['drum'])].append(ang(r['x']))
    # pretest
    hands, items, z, _ = L.A.raw_features(PRE_RAW, L.DET); raw = L.A.signals(hands)
    ev = [e for e in L.legacy.midi(L.MID/'測試1.mid')['events'] if e['vel'] >= L.GHOST_VEL]
    pk = np.sort([p['t'] for p in raw['peaks'] if p['value'] >= .5])
    off, rate = L.align(np.array([e['t'] for e in ev]), pk)
    for e in ev: e['csv_t'] = e['t']+off
    for e, c in match(ev, items):
        if c['peak']['hand'] == 'L': stats[('前測', e['name'])].append(ang(c['x']))
    print('pretest offset', round(off), 'align', round(rate, 3), 'zero', {h: [round(v, 1) for v in z[h][:2]] for h in z})
    # today's single-hand, left hand target drum + right-hand single takes not needed
    for name, (drum, rn) in L.SINGLE.items():
        if not name.endswith('L'): continue
        s = L.session(name, rn)
        tg = [e for e in s['events'] if e['name'] == drum and e['vel'] >= L.GHOST_VEL]
        sn = [e for e in s['events'] if e['name'] == '小鼓' and e['vel'] >= L.GHOST_VEL and e['t'] < tg[0]['t']]
        a_sn = np.array([ang(c['x']) for e, c in match(sn, s['items'], 'L')]); a_tg = np.array([ang(c['x']) for e, c in match(tg, s['items'], 'L')])
        for v in a_tg: stats[('單手', drum)].append(tuple(v))
        if len(a_sn): stats[('單手Δ', drum)] = list(a_tg-np.median(a_sn, 0))
        print(name, 'zero', [round(v, 1) for v in s['zero']['L'][:2]], 'L snare anchors', len(a_sn))
    groups = ['R4', 'R5', 'R6', 'R7', '前測', '單手']
    res = {}
    print('\n左手 每鼓 擊中瞬間姿勢（相對鼓面歸零）: 偏航 yaw / 俯仰 pitch (度), n')
    for d in DR:
        line = []
        for g in groups:
            a = np.array(stats.get((g, d), []))
            if len(a): line.append(f'{g} {np.median(a[:,0]):6.1f}/{np.median(a[:,1]):6.1f} n{len(a)}'); res[f'{g}|{d}'] = [float(np.median(a[:, 0])), float(np.median(a[:, 1])), len(a)]
            else: line.append(f'{g}  -')
        print(f'{d:8s} | ' + ' | '.join(line))
    print('\n相對同場小鼓的差（drum − 小鼓）: Δyaw/Δpitch')
    for d in DR[1:]:
        line = []
        for g in ['R4', 'R5', 'R6', 'R7', '前測']:
            a, b = np.array(stats.get((g, d), [])), np.array(stats.get((g, '小鼓'), []))
            if len(a) and len(b):
                dd = np.median(a, 0)-np.median(b, 0); line.append(f'{g} {dd[0]:6.1f}/{dd[1]:6.1f}'); res[f'Δ{g}|{d}'] = dd.tolist()
            else: line.append(f'{g}  -')
        a = np.array(stats.get(('單手Δ', d), []))
        if len(a): dd = np.median(a, 0); line.append(f'今天單手 {dd[0]:6.1f}/{dd[1]:6.1f}'); res[f'Δ單手|{d}'] = dd.tolist()
        print(f'{d:8s} | ' + ' | '.join(line))
    (HERE/'diag_pretest_left.json').write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
