"""Re-check the takes flagged by screen_takes.py through the real field engine (with the zero-direction
guard), each with the leave-one-day-out model of its own day. Also reports the automatic anchor decision.

  python screen_field.py [--below 0.5] [--prefix lodoex]   -> out/screen_field.json
"""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--below', type=float, default=0.5)
ap.add_argument('--prefix', default='lodoex')
ap.add_argument('--code', default='第十次的包_暫定')
ARGS = ap.parse_args() if __name__ == '__main__' else ap.parse_args([])

import joblib

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT/ARGS.code/'Program'), str(HERE)]
import seventh_core as core
import rounds_data as RD
import verify_package_1027 as VP
L5, K = RD.L5, RD.K


def raw_of(name, T):
    if name.startswith('eighth_'): return L5.r8_docs()[name]['raw_csv']
    for R, c in RD.config().items():
        if name.startswith(f'{R}:'): return RD.docs_of(R, c)[name]['raw_csv']
    return VP.V.C['sessions'][name]['raw']


def main():
    rows, names, Lb, T, info = RD.setup(); _, config = core.load_components()
    flagged = {k: v for k, v in json.loads((RD.OUT/'screen_takes.json').read_text(encoding='utf-8')).items() if v['rate'] < ARGS.below}
    res = {}
    for name, v in sorted(flagged.items(), key=lambda kv: kv[1]['rate']):
        D = v['day']; b = joblib.load(RD.OUT/f'model_{ARGS.prefix}_{D}.joblib'); core.runtime.attach_boosters(b)
        raw = raw_of(name, T); hands = core.trajectory.read_raw(raw)
        with open(raw, encoding='utf-8-sig', newline='') as f: source = list(csv.DictReader(f))
        eng = core.Engine(b, config, snare_anchor=core.runtime.PRETEST_SNARE_ANCHOR); P = []
        for _, h, s in sorted((s['row_index'], h, s) for h, a in hands.items() for s in a):
            row = source[s['row_index']]
            p, _ = eng.push(h, s, dict(ax=s['ax'], ay=s['ay'], az=s['az'], yaw=float(row['yaw_deg']), pitch=float(row['pitch_deg']), roll=s['roll']))
            P.extend(p)
        P.sort(key=lambda p: (p['peak_ms'], p['hand']))
        used, excl = T[name] if name in T else L5.F.truth_of(name)
        ok, n, cells = VP.score(used, P); rt = eng.runtime
        res[name] = dict(day=D, research=v['rate'], field=round(ok/max(n, 1), 4), ok=ok, n=n,
                         anchored={h: rt.anchored[h] for h in 'LR'},
                         anchored_drum={h: (rt.classes[rt.anchored_drum[h]] if rt.anchored_drum[h] is not None else None) for h in 'LR'},
                         zero_rot={h: float(rt.zero_rot[h]) for h in 'LR'},
                         cells={k: f'{c[0]}/{c[1]} {c[2]}' for k, c in sorted(cells.items())})
        print(json.dumps(dict(name=name, **res[name]), ensure_ascii=False), flush=True)
    (RD.OUT/'screen_field.json').write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
