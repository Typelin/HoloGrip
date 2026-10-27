"""Auto-label the 10/27 recordings (8th session). Offline annotation; raw/MIDI originals untouched.

Single-hand reinforcement (10 takes): protocol rule.
  * opening snares = anchor; hand from glove (nearest >=1.5 g peak), unresolved -> neighbours' hand
  * target drum notes -> the take's hand
  * any other note, or velocity < 10 -> exclude (ghost / crosstalk / accidental)
Four-drum (測試3改) and song (測試2改): same flow as round 7.
  * crosstalk ghost (vel < 20 within 30 ms of a >=3x stronger note of another drum) -> exclude
  * < 2.1 g max of both hands within +-60 ms -> exclude: unclear (detector never fires)
  * suspected ghost (vel < 35% of drum median and both hands < 0.8 g) -> exclude
  * conservative IMU evidence (algorithm_imu_rederived_v1) -> label
  * hand auto-labeler v2 (two XGBoost models agree, confidence >= 0.98) -> label
  * otherwise -> review (blank), with the v2 suggestion
Pretest (測試1) is never labelled for training.
"""
from __future__ import annotations
import csv, json, sys, warnings
from collections import Counter
from pathlib import Path
import numpy as np
warnings.filterwarnings('ignore')
ROOT = Path(__file__).resolve().parents[2]
P6 = ROOT/'第六次的包_暫定2026-10-16'; P7 = ROOT/'第七次的包_暫定2026-10-20'; P8 = ROOT/'第八次的包_暫定2026-10-27'
sys.path[:0] = [str(P7/'Program'), str(P6/'Program'), str(ROOT/'Research/AlignmentAudit_20261020'), str(ROOT/'Research/AutoLabel_v2_20261021')]
import joblib
import auto_label_experiment as A
import rederive_sixth_1016 as legacy
from build_sixth_independent_review_1016 import evidence
from hand_autolabel_v2 import fit, features, HIST

MID = Path(r'C:/Users/Typelin_Station/Desktop/今天'); RAW = P8/'Data/Raw'
OUT = P8/'Data/自動標記_20261027'; OUT.mkdir(parents=True, exist_ok=True)
SINGLE = {'crash L': ('Crash', 'S20261020_L_Crash_take02_raw_100hz_20261027_134706'),
          'crash R': ('Crash', 'S20261020_R_Crash_take01_raw_100hz_20261027_135005'),
          'fioor ton L': ('落地 Tom', 'S20261020_P01_L_GND_take02_raw_100hz_20261027_131305'),
          'fioor ton R': ('落地 Tom', 'S20261020_P01_R_GND_take01_raw_100hz_20261027_131706'),
          'ride L': ('Ride', 'S20261020_P01_L_Ride_take01_raw_100hz_20261027_130314'),
          'ride R': ('Ride', 'S20261020_P01_R_Ride_take01_raw_100hz_20261027_130814'),
          'ton1 L': ('高音 Tom', 'S20261020_P01_L_HI_take01_raw_100hz_20261027_133332'),
          'ton1 R': ('高音 Tom', 'S20261020_P01_R_HI_take01_raw_100hz_20261027_133640'),
          'ton2 L': ('中音 Tom', 'S20261020_P01_L_MID_take01_raw_100hz_20261027_132126'),
          'ton2 R': ('中音 Tom', 'S20261020_P01_R_MID_take02_raw_100hz_20261027_132946')}
SONGS = {'測試3改': ('four_drum', 'S20261020_P01_song02_take02_raw_100hz_20261027_123510'),
         '測試2改': ('song', 'S20261020_P01_song01_take01_raw_100hz_20261027_115924')}
GHOST_VEL, UNCLEAR_G, AUTO_CONF = 10, 2.1, 0.98
DET = None


def align(T, P):
    """Offset (csv = midi + off) maximising matches within 50 ms; histogram candidates, 5 ms search, median refine."""
    def score(off):
        i = np.clip(np.searchsorted(P, T+off), 1, len(P)-1)
        return int((np.minimum(np.abs(P[i]-T-off), np.abs(P[i-1]-T-off)) <= 50).sum())
    diff = (P[None, :]-T[:, None]).ravel()
    h, e = np.histogram(diff, bins=np.arange(diff.min(), diff.max()+40, 40))
    best = max((score(o), o) for k in np.argsort(-h)[:15] for o in np.arange(e[k]-40, e[k]+80, 5.0))
    off = best[1]; i = np.clip(np.searchsorted(P, T+off), 1, len(P)-1)
    r = np.where(np.abs(P[i]-T-off) < np.abs(P[i-1]-T-off), P[i]-T-off, P[i-1]-T-off)
    return off+float(np.median(r[np.abs(r) <= 50])), best[0]/len(T)


def session(name, raw_name):
    hands, items, z, diag = A.raw_features(RAW/f'{raw_name}.csv', DET)
    raw = A.signals(hands)
    ev = legacy.midi(MID/f'{name}.mid')['events']
    pk = np.sort([p['t'] for p in raw['peaks'] if p['value'] >= .5])
    off, rate = align(np.array([e['t'] for e in ev if e['vel'] >= GHOST_VEL]), pk)
    for e in ev: e['csv_t'] = e['t']+off
    return dict(name=name, raw=str(RAW/f'{raw_name}.csv'), events=ev, items=items, raw_signals=raw, zero=z, offset_ms=off, align_rate=rate)


def hand_max(raw, t, w=60):
    out = {}
    for h in 'LR':
        a = np.asarray(raw['hands'][h]); m = (a[:, 0] >= t-w) & (a[:, 0] <= t+w)
        out[h] = float(a[m, 1].max())+1 if m.any() else 0.  # stored as |a|-1 -> back to g
    return out


def label_single(name, drum, raw_name):
    s = session(name, raw_name); hand = name[-1]; ev = s['events']; raw = s['raw_signals']
    pk = {h: np.array(sorted(p['t'] for p in raw['peaks'] if p['hand'] == h and p['value'] >= .5)) for h in 'LR'}
    first_target = min(e['t'] for e in ev if e['name'] == drum and e['vel'] >= GHOST_VEL)
    labels = {}; anchors = []
    for e in ev:
        k = str(e['id'])
        if e['vel'] < GHOST_VEL:
            labels[k] = dict(mark='exclude', drum=e['name'], source='protocol_single_hand_1027', rule='ghost_low_velocity', comment=f"力度 {e['vel']}")
        elif e['name'] == drum:
            labels[k] = dict(mark=hand, drum=drum, source='protocol_single_hand_1027', rule='single_hand_target', comment='')
        elif e['name'] == '小鼓' and e['t'] < first_target:
            d = {h: (np.abs(pk[h]-e['csv_t']).min() if len(pk[h]) else 1e9) for h in 'LR'}
            anchors.append((e, min(d, key=d.get) if min(d.values()) <= 60 else '?'))
        else:
            labels[k] = dict(mark='exclude', drum=e['name'], source='protocol_single_hand_1027', rule='other_drum_ghost_or_crosstalk', comment='單手段落非目標鼓')
    # anchors are an L block then an R block: an unresolved one takes its neighbours' hand when they agree
    seq = [g for _, g in anchors]
    for i, g in enumerate(seq):
        if g == '?':
            prev = next((seq[j] for j in range(i-1, -1, -1) if seq[j] != '?'), None)
            nxt = next((seq[j] for j in range(i+1, len(seq)) if seq[j] != '?'), None)
            seq[i] = prev if (prev == nxt or nxt is None) else (nxt if prev is None else '?')
    for (e, g0), g in zip(anchors, seq):
        labels[str(e['id'])] = (dict(mark=g, drum='小鼓', source='protocol_single_hand_1027',
                                     rule='anchor_snare_glove_hand' if g0 != '?' else 'anchor_snare_neighbour_hand', comment='')
                                if g in ('L', 'R') else
                                dict(mark='review', drum='小鼓', source='protocol_single_hand_1027', rule='anchor_snare_unresolved', comment=''))
    return s, labels


def label_song(name, raw_name, models, pose_free, names):
    s = session(name, raw_name); raw = s['raw_signals']; ev_all = s['events']
    cons = evidence(ev_all, raw)
    ev, X, nm = features(s); assert nm == names
    pa = np.mean([m.predict_proba(X)[:, 1] for m in models['all']], 0)
    pn = np.mean([m.predict_proba(X[:, pose_free])[:, 1] for m in models['no_pose']], 0)
    conf = np.minimum(np.maximum(pa, 1-pa), np.maximum(pn, 1-pn)); agree = (pa >= .5) == (pn >= .5); v2 = np.where(pa+pn >= 1, 'L', 'R')
    ix = {n: i for i, n in enumerate(names)}
    labels = {}; rows = []
    for k, e in enumerate(ev):
        key = str(e['id']); mx = hand_max(raw, e['csv_t']); gmax = max(mx.values())
        ghost = X[k, ix['vel_rel']] < .35 and max(X[k, ix['L_max60']], X[k, ix['R_max60']]) < .8
        c = cons.get(key)
        # crosstalk: a weak note riding on a >=3x stronger note of another drum within 30 ms (same physical hit)
        xtalk = e['vel'] < 20 and any(q is not e and abs(q['csv_t']-e['csv_t']) <= 30 and q['vel'] >= 3*e['vel'] for q in ev[max(0, k-4):k+5])
        if xtalk:
            lab = dict(mark='exclude', drum=e['name'], source='algorithm_autolabel_v3', rule='crosstalk_ghost', comment=f"力度 {e['vel']}，30 ms 內有強 3 倍的另一鼓")
        elif gmax < UNCLEAR_G:
            lab = dict(mark='exclude', drum=e['name'], source='algorithm_autolabel_v3', rule='unclear_below_2p1g', comment=f'兩手最大 {gmax:.2f} g')
        elif ghost:
            lab = dict(mark='exclude', drum=e['name'], source='algorithm_autolabel_v3', rule='suspected_ghost', comment=f"力度 {e['vel']}")
        elif c:
            lab = dict(c, comment=f'v2 {v2[k]} {conf[k]:.3f}')
        elif agree[k] and conf[k] >= AUTO_CONF:
            lab = dict(mark=str(v2[k]), drum=e['name'], source='algorithm_autolabel_v2', human_confirmed=False, rule='v2_agree_conf', comment=f'{conf[k]:.4f}')
        else:
            lab = dict(mark='review', drum=e['name'], source='algorithm_autolabel_v2', rule='needs_review',
                       comment=f'建議 {v2[k]} 信心 {conf[k]:.3f}' + ('' if agree[k] else ' 兩模型不一致'))
        labels[key] = lab
        rows.append(dict(id=e['id'], midi_ms=round(e['t']), csv_ms=round(e['csv_t']), drum=e['name'], vel=e['vel'], L_g=round(mx['L'], 2), R_g=round(mx['R'], 2),
                         v2_hand=str(v2[k]), v2_conf=round(float(conf[k]), 4), models_agree=bool(agree[k]),
                         conservative=c['mark'] if c else '', mark=lab['mark'], rule=lab['rule']))
    for e in ev_all:
        labels.setdefault(str(e['id']), dict(mark='exclude', drum=e['name'], source='algorithm_autolabel_v3', rule='unmapped', comment=''))
    return s, labels, rows


def training_rows(s, labels):
    """Same ownership rule as soft_hit_experiment.rows_from_labels: labelled event <-> detected peak of that hand within 60 ms."""
    ev = {str(e['id']): e for e in s['events']}; edges = []
    for k, m in labels.items():
        if m['mark'] not in ('L', 'R'): continue
        for j, c in enumerate(s['items']):
            p = c['peak']
            if p['hand'] == m['mark'] and abs(p['peak_ms']-ev[k]['csv_t']) <= 60: edges.append((abs(p['peak_ms']-ev[k]['csv_t']), k, j))
    own, used = {}, set()
    for d, k, j in sorted(edges):
        if k not in own and j not in used: own[k] = j; used.add(j)
    return own


def write(s, labels, kind, extra=None):
    own = training_rows(s, labels); ev = s['events']
    doc = dict(schema='hologrip-8th-autolabel-1027-v1', session=s['name'], kind=kind, raw_csv=s['raw'], midi=str(MID/f"{s['name']}.mid"),
               midi_to_csv_offset_ms=round(s['offset_ms'], 1), align_rate=round(s['align_rate'], 4), labels=labels,
               events=[dict(id=e['id'], ordinal=e['ordinal'], midi_ms=round(e['t'], 1), csv_ms=round(e['csv_t'], 1), note=e['note'], vel=e['vel'], drum=e['name'],
                            training_peak_ms=s['items'][own[str(e['id'])]]['peak']['peak_ms'] if str(e['id']) in own else None) for e in ev],
               policy='pretest never labelled; exclude = ghost/crosstalk/unclear; review = blank for human')
    (OUT/f"{s['name']}_自動標記_20261027.json").write_text(json.dumps(doc, ensure_ascii=False, indent=1, default=float), encoding='utf-8')
    if extra:
        with open(OUT/f"{s['name']}_逐筆_20261027.csv", 'w', encoding='utf-8-sig', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(extra[0])); w.writeheader(); w.writerows(extra)
    hd = Counter((labels[str(e['id'])]['mark'], e['name']) for e in ev if str(e['id']) in own)
    return dict(session=s['name'], kind=kind, events=len(ev), offset_ms=round(s['offset_ms'], 1), align_rate=round(s['align_rate'], 4),
                marks=dict(Counter(m['mark'] for m in labels.values())), rules=dict(Counter(m['rule'] for m in labels.values())),
                training_hits=len(own), training_by_hand_drum={f'{h}|{d}': n for (h, d), n in sorted(hd.items())})


def main():
    global DET
    C = joblib.load(P7/'Data/實驗_姿態模型自動標記_20261020/實驗快取.joblib'); DET = C['detector']
    summary = []
    for name, (drum, rn) in SINGLE.items():
        s, labels = label_single(name, drum, rn); sm = write(s, labels, 'single_hand'); summary.append(sm)
        print(json.dumps(sm, ensure_ascii=False), flush=True)
    D = joblib.load(ROOT/'Research/AutoLabel_v2_20261021/out/loro.joblib'); data, names = D['data'], D['names']
    pose_free = np.array([i for i, n in enumerate(names) if '_rot' not in n]); allc = np.arange(len(names))
    Xtr = np.vstack([data[n]['X'][data[n]['y'] >= 0] for n in HIST]); ytr = np.concatenate([data[n]['y'][data[n]['y'] >= 0] for n in HIST])
    models = {k: [fit(Xtr[:, c], ytr, sd) for sd in range(3)] for k, c in (('all', allc), ('no_pose', pose_free))}
    for name, (kind, rn) in SONGS.items():
        s, labels, rows = label_song(name, rn, models, pose_free, names); sm = write(s, labels, kind, rows); summary.append(sm)
        print(json.dumps(sm, ensure_ascii=False), flush=True)
    (OUT/'總表_20261027.json').write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
