"""Full-pipeline exam of the new model on EVERY round 4-7 recording (training ones included, reported as
resubstitution). Raw -> detector 2.1 g -> per-hit causal features -> new committee -> MIDI.

Per recording:
  MIDI used   : events with a valid answer (L/R label; pretests: all evaluable MIDI)
  detected    : model outputs inside the MIDI span
  matched     : outputs paired one-to-one with a used MIDI within 80 ms (time only)
  correct     : matched, drum right, and hand right where a hand answer exists
  wrong drum / wrong hand, missed (MIDI never matched), extra (outputs not near any MIDI,
  including excluded/unlabelled MIDI; outputs near excluded MIDI are counted separately)
Per hand x drum (recordings with hand answers): MIDI count, accuracy, most common mistake.
"""
import json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np, joblib
from scipy.optimize import linear_sum_assignment
import common as K
import soft_hit_experiment as S
import venue_free_runtime_1020 as rt
import causal_hit_detector_1016 as detector
import seventh_core as core

ORDER = [('fourth_song03', '第4次 四鼓強化', 'train'), ('pretest:round4', '第4次 前測', 'test'),
         ('fifth_song01', '第5次 歌曲1', 'train'), ('fifth_song02', '第5次 歌曲2', 'train'), ('fifth_song03', '第5次 四鼓強化', 'train'),
         ('pretest:round5', '第5次 前測', 'test'),
         ('sixth_S02', '第6次 歌曲2', 'train'), ('sixth_S08', '第6次 四鼓強化', 'train'), ('sixth_S04', '第6次 單手 左Hi-Hat', 'train'),
         ('sixth_S05', '第6次 單手 左Crash', 'train'), ('sixth_S06', '第6次 單手 左Ride', 'train'), ('sixth_S07', '第6次 單手 右Crash', 'train'),
         ('pretest:round6', '第6次 前測', 'test'),
         ('seventh_01', '第7次 前測1 原鼓位', 'test'), ('seventh_03', '第7次 歌曲1 原鼓位', 'train'), ('seventh_04', '第7次 歌曲2 原鼓位', 'train'),
         ('seventh_05', '第7次 四鼓強化 原鼓位', 'train'), ('seventh_06', '第7次 單手 左高音Tom', 'train'), ('seventh_07', '第7次 單手 左中音Tom', 'train'),
         ('seventh_08', '第7次 單手 左落地Tom', 'train'), ('seventh_09', '第7次 單手 右高音Tom', 'train'), ('seventh_10', '第7次 單手 左Crash', 'train'),
         ('seventh_11', '第7次 單手 左小鼓', 'train'), ('seventh_12', '第7次 前測2 移動後', 'test'), ('seventh_13', '第7次 歌曲1 移動後', 'train'),
         ('seventh_14', '第7次 歌曲2 移動後', 'train'), ('seventh_15', '第7次 前測3 移動後', 'test')]
OLD = {s['name']: s for s in joblib.load(K.P6/'Data/鼓面歸零_跨次完整CSV考試_20261019_v1/特徵快取.joblib')}


def truth_of(name):
    """(used: list of dict(t, drum, hand|None), excluded_times)"""
    C = S.C
    if name in K.TESTS:
        return [dict(t=float(g['t']), drum=g['drum'], hand=None) for g in C['tests'][name]['truth']], []
    s = C['sessions'][name]; ev = {str(e['id']): e for e in s['events']}
    if name.startswith('seventh_') and name[-2:] in ('03', '04', '05', '13', '14'):
        L = json.loads((K.LAB/f'{name[-2:]}_完成標記_20261021.json').read_text(encoding='utf-8'))['labels']
        lab = {k: (v['mark'], v['drum']) for k, v in L.items()}
    elif name.startswith('seventh_'):
        lab = {str(r['id']): (r['hand'], r['drum']) for r in C['auto'] if r['session'] == name}
    else:
        b = json.loads(Path(OLD[name]['backup']).read_text(encoding='utf-8-sig')) if 'backup' in OLD[name] else None
        if b and 'labels' in b:
            lab = {k: (v.get('mark'), v.get('drum') or ev.get(k, {}).get('name')) for k, v in b['labels'].items() if not v.get('needs_recheck')}
        else:
            lab = {k: (v['hand'], v['drum']) for k, v in s['reference'].items()}
    evs = sorted(ev.values(), key=lambda e: e['t']); ghost = set()
    for i, a in enumerate(evs):
        if a.get('vel', 99) < 20 and any(b is not a and b['note'] == a['note'] and (110 <= a['t']-b['t'] <= 260 or 0 <= a['t']-b['t'] <= 20) for b in evs[max(0, i-8):i+1]):
            ghost.add(str(a['id']))
    used, excl = [], []
    for k, e in ev.items():
        if k in ghost: excl.append(float(e['csv_t'])); continue
        m = lab.get(k)
        if m and m[0] in ('L', 'R'): used.append(dict(t=float(e['csv_t']), drum=m[1], hand=m[0]))
        elif e.get('name'): excl.append(float(e['csv_t']))
    return used, excl


def detect(raw, bundle, cfg):
    hands = rt.world.read_raw(raw); prep = rt.world.base.legacy.prepare(hands)
    peaks, _ = detector.detect(hands, cfg); out = []
    for p in peaks:
        s = prep[p['hand']]['rows'][p['available_index']]
        x = rt.feature(prep, p, (s['absolute_yaw']-s['yaw']+180) % 360-180, s['absolute_pitch']-s['pitch'])
        if x is None: continue
        q = rt.probabilities(bundle, x[None, :], p['hand'])[0]
        out.append(dict(t=float(p['peak_ms']), hand=p['hand'], drum=bundle['classes'][int(q.argmax())], score=float(q.max())))
    return out


def score(used, excl, preds):
    tt = np.array([g['t'] for g in used]); lo, hi = tt.min()-200, tt.max()+200
    P = [p for p in preds if lo <= p['t'] <= hi]
    pt = np.array([p['t'] for p in P]); dt = np.abs(tt[:, None]-pt[None, :])
    hm = np.array([[g['hand'] is not None and g['hand'] != p['hand'] for p in P] for g in used]) if len(P) else np.zeros((len(used), 0), bool)
    base = np.where(dt <= 80, dt+300*hm, 1e9)
    cost = np.hstack([base, np.full((len(tt), len(tt)), 1e6)])
    r, c = linear_sum_assignment(cost); pairs = {i: j for i, j in zip(r, c) if j < len(pt) and cost[i, j] <= 380}
    res = Counter(); cells = []
    for i, g in enumerate(used):
        j = pairs.get(i)
        if j is None: res['missed'] += 1; cells.append((g, 'missed')); continue
        p = P[j]
        if g['hand'] and p['hand'] != g['hand']: res['wrong_hand'] += 1; cells.append((g, f"判成{p['hand']}手"))
        elif p['drum'] != g['drum']: res['wrong_drum'] += 1; cells.append((g, p['drum']))
        else: res['correct'] += 1; cells.append((g, 'ok'))
    unmatched = [P[j] for j in range(len(P)) if j not in set(pairs.values())]
    ex = np.array(excl) if excl else np.array([1e18])
    near_ex = sum(1 for p in unmatched if np.min(np.abs(ex-p['t'])) <= 80)
    res.update(midi=len(used), detected=len(P), matched=len(pairs), extra=len(unmatched)-near_ex, near_excluded=near_ex)
    return dict(res), cells


def main():
    new = joblib.load(K.OUT/'venue_free_committee_rounds4to7_20261021.joblib'); new['classes'] = [str(c) for c in new['classes']]
    rt.attach_boosters(new); _, cfg = core.load_components()
    rows, cellstat = [], defaultdict(Counter)
    for name, title, role in ORDER:
        used, excl = truth_of(name)
        raw = S.C['tests'][name]['raw'] if name in K.TESTS else S.C['sessions'][name]['raw']
        res, cells = score(used, excl, detect(raw, new, cfg))
        res.update(name=name, title=title, role=role); rows.append(res)
        for g, o in cells:
            if g['hand']: cellstat[(g['hand'], g['drum'])][o] += 1
        print(title, res, flush=True)
    cell_rows = []
    for h in 'LR':
        for d in K.DRUMS:
            c = cellstat[(h, d)]; n = sum(c.values()); mist = Counter({k: v for k, v in c.items() if k != 'ok'})
            drum_m = Counter({k: v for k, v in mist.items() if k in K.DRUMS})
            cell_rows.append(dict(hand=h, drum=d, midi=n, correct=c['ok'], acc=round(c['ok']/max(n, 1), 4), missed=c['missed'],
                                  wrong_hand=sum(v for k, v in c.items() if k.startswith('判成')), wrong_drum=sum(drum_m.values()), top_drum=drum_m.most_common(2)))
            print(h, d, cell_rows[-1], flush=True)
    (K.OUT/'full_report.json').write_text(json.dumps(dict(recordings=rows, cells=cell_rows), ensure_ascii=False, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
