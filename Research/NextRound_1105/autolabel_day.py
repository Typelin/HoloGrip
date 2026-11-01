"""Auto-label one new field day with the 10/27 rules (Research/AutoLabel_1027/autolabel_1027.py, unchanged).

Single-hand takes: protocol rule (opening snares = anchor with glove hand; target drum = the take's
hand; everything else excluded). Songs / four-drum: crosstalk, < 2.1 g and suspected ghosts excluded;
conservative IMU evidence or hand labeller v2 (confidence >= 0.98) label; the rest is left for review.
The pretest is never labelled. Raw / MIDI originals are only read.

  python autolabel_day.py R9   -> <package>/Data/自動標記_<date>/*_自動標記_<date>.json, 總表_<date>.json
"""
from __future__ import annotations
import json, os, sys
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


def main():
    c = json.loads((HERE/os.environ.get('HOLOGRIP_ROUNDS', 'rounds.json')).read_text(encoding='utf-8'))[DAY]
    date = c['date']; out = label_dir(c); out.mkdir(parents=True, exist_ok=True)
    AL.MID = Path(c['midi_dir']); AL.RAW = ROOT/c['package']/'Data/Raw'; AL.OUT = out
    AL.DET = joblib.load(AL.P7/'Data/實驗_姿態模型自動標記_20261020/實驗快取.joblib')['detector']
    for name in c['single']:
        assert name[-1] in 'LR', f'single-hand MIDI name must end with L or R: {name}'
    summary = []
    for name, (drum, raw) in c['single'].items():
        s, labels = AL.label_single(name, drum, raw); summary.append(AL.write(s, labels, 'single_hand'))
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
