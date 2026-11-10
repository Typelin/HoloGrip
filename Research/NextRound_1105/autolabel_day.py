"""Auto-label one new field day with the 10/27 rules (Research/AutoLabel_1027/autolabel_1027.py, unchanged).

Single-hand takes: protocol rule (opening snares = anchor with glove hand; target drum = the take's
hand; everything else excluded). Songs / four-drum: crosstalk, < 2.1 g and suspected ghosts excluded;
conservative IMU evidence or hand labeller v2 (confidence >= 0.98) label; the rest is left for review.
The pretest is never labelled. Raw / MIDI originals are only read.

  python autolabel_day.py R9   -> <package>/Data/自動標記_<date>/*_自動標記_<date>.json, 總表_<date>.json
"""
from __future__ import annotations
import json, os, sys, unicodedata
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DAY = sys.argv[1] if len(sys.argv) > 1 else 'R9'
sys.path[:0] = [str(ROOT/'Research/AutoLabel_1027'), str(HERE)]
import joblib
import autolabel_1027 as AL


def label_dir(c):
    if c.get('label_dir'): return Path(c['label_dir']) if Path(c['label_dir']).is_absolute() else HERE/c['label_dir']
    return ROOT/c['package']/'Data'/f"自動標記_{c['date']}"


def hand_of(name):
    """Take hand from the MIDI name's last character (full-width Ｌ／Ｒ accepted)."""
    h = unicodedata.normalize('NFKC', name).strip()[-1]
    assert h in 'LR', f'single-hand MIDI name must end with L or R: {name}'
    return h


def label_single(name, drums, raw_name):
    """autolabel_1027.label_single for a take with one or more target drums (e.g. 中音 Tom then Ride):
    opening snares before the first target note = anchor (glove hand); target notes = the take's hand;
    any other note or velocity < 10 = excluded."""
    s = AL.session(name, raw_name); hand = hand_of(name); ev = s['events']; raw = s['raw_signals']
    pk = {h: np.array(sorted(p['t'] for p in raw['peaks'] if p['hand'] == h and p['value'] >= .5)) for h in 'LR'}
    first_target = min(e['t'] for e in ev if e['name'] in drums and e['vel'] >= AL.GHOST_VEL)
    labels = {}; anchors = []
    for e in ev:
        k = str(e['id'])
        if e['vel'] < AL.GHOST_VEL:
            labels[k] = dict(mark='exclude', drum=e['name'], source='protocol_single_hand_day', rule='ghost_low_velocity', comment=f"力度 {e['vel']}")
        elif e['name'] in drums:
            labels[k] = dict(mark=hand, drum=e['name'], source='protocol_single_hand_day', rule='single_hand_target', comment='')
        elif e['name'] == '小鼓' and e['t'] < first_target:
            d = {h: (np.abs(pk[h]-e['csv_t']).min() if len(pk[h]) else 1e9) for h in 'LR'}
            anchors.append((e, min(d, key=d.get) if min(d.values()) <= 60 else '?'))
        else:
            labels[k] = dict(mark='exclude', drum=e['name'], source='protocol_single_hand_day', rule='other_drum_ghost_or_crosstalk', comment='單手段落非目標鼓')
    seq = [g for _, g in anchors]
    for i, g in enumerate(seq):
        if g == '?':
            prev = next((seq[j] for j in range(i-1, -1, -1) if seq[j] != '?'), None)
            nxt = next((seq[j] for j in range(i+1, len(seq)) if seq[j] != '?'), None)
            seq[i] = prev if (prev == nxt or nxt is None) else (nxt if prev is None else '?')
    for (e, g0), g in zip(anchors, seq):
        labels[str(e['id'])] = (dict(mark=g, drum='小鼓', source='protocol_single_hand_day',
                                     rule='anchor_snare_glove_hand' if g0 != '?' else 'anchor_snare_neighbour_hand', comment='')
                                if g in ('L', 'R') else
                                dict(mark='review', drum='小鼓', source='protocol_single_hand_day', rule='anchor_snare_unresolved', comment=''))
    return s, labels


def main():
    c = json.loads((HERE/os.environ.get('HOLOGRIP_ROUNDS', 'rounds.json')).read_text(encoding='utf-8'))[DAY]
    date = c['date']; out = label_dir(c); out.mkdir(parents=True, exist_ok=True)
    AL.MID = Path(c['midi_dir']); AL.RAW = ROOT/c['package']/'Data/Raw'; AL.OUT = out
    AL.DET = joblib.load(AL.P7/'Data/實驗_姿態模型自動標記_20261020/實驗快取.joblib')['detector']
    for name in c['single']: hand_of(name)
    summary = []
    for name, (drum, raw) in c['single'].items():
        s, labels = label_single(name, [drum] if isinstance(drum, str) else list(drum), raw); summary.append(AL.write(s, labels, 'single_hand'))
        print(json.dumps(summary[-1], ensure_ascii=False), flush=True)
    if c['songs']:
        D = joblib.load(ROOT/'Research/AutoLabel_v2_20261021/out/loro.joblib'); data, names = D['data'], D['names']
        pose_free = np.array([i for i, n in enumerate(names) if '_rot' not in n]); allc = np.arange(len(names))
        Xtr = np.vstack([data[n]['X'][data[n]['y'] >= 0] for n in AL.HIST]); ytr = np.concatenate([data[n]['y'][data[n]['y'] >= 0] for n in AL.HIST])
        models = {k: [AL.fit(Xtr[:, cc], ytr, sd) for sd in range(3)] for k, cc in (('all', allc), ('no_pose', pose_free))}
        for name, (kind, raw) in c['songs'].items():
            s, labels, rows = AL.label_song(name, raw, models, pose_free, names); summary.append(AL.write(s, labels, kind, rows))
            print(json.dumps(summary[-1], ensure_ascii=False), flush=True)
    # autolabel_1027.write names files with the 10/27 date; rename to this day
    for p in list(out.glob('*_20261027.*')):
        p.replace(p.with_name(p.name.replace('_20261027.', f'_{date}.')))
    for p in out.glob(f'*_自動標記_{date}.json'):
        d = json.loads(p.read_text(encoding='utf-8')); d['day'] = DAY; d['date'] = date
        p.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding='utf-8')
    (out/f'總表_{date}.json').write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding='utf-8')
    review = sum(s['marks'].get('review', 0) for s in summary)
    print('labelled takes', len(summary), 'training hits', sum(s['training_hits'] for s in summary), 'left for review', review, flush=True)


if __name__ == '__main__':
    main()
