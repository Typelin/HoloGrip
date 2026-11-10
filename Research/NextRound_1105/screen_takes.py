"""Screen every labelled training take (rounds 4-9) for inconsistent data.

Each take is scored by the leave-one-day-out model of its own day (out/model_<prefix>_<day>.joblib), so the
model has never seen the take or its day. Research batch decoder (eval_all_9th.predict: committee, per-drum
offset correction, automatic anchor, position prior, pose continuity; no zero-direction guard).
A very low score means the take disagrees with every other day: wrong labels, a shifted kit without a
reference, or a drumhead zero far off. Per hand, and the take's median hit pose per drum vs the model's.

  python screen_takes.py [--prefix lodoex]   -> out/screen_takes.json / .md
"""
from __future__ import annotations
import argparse, json, sys
from collections import defaultdict
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--prefix', default='lodoex')
ARGS = ap.parse_args() if __name__ == '__main__' else ap.parse_args([])

import numpy as np
import joblib

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1]/'Research/AutoLabel_1027')]
import eval_all_9th as A   # first: it pins the 9th package code (same runtime as the 10th)
import rounds_data as RD
L5, K, H, F = RD.L5, RD.K, RD.H, A.F
core = A.core


def main():
    rows, names, Lb, T, info = RD.setup()
    bundles = {}; res = {}
    for name in sorted([n for n in names if n not in K.TESTS], key=lambda n: (L5.day_of(n), n)):
        D = L5.day_of(name)
        if D not in bundles:
            b = joblib.load(RD.OUT/f'model_{ARGS.prefix}_{D}.joblib'); core.runtime.attach_boosters(b); bundles[D] = b
        P = A.predict(bundles[D], name)
        used, excl = T[name] if name in T else F.truth_of(name)
        r, cl = F.score(used, excl, P)
        hand = defaultdict(lambda: [0, 0]); cells = defaultdict(lambda: [0, 0])
        for g, o in cl:
            hand[g['hand']][0] += o == 'ok'; hand[g['hand']][1] += 1
            cells[f"{g['hand']}|{g['drum']}"][0] += o == 'ok'; cells[f"{g['hand']}|{g['drum']}"][1] += 1
        res[name] = dict(day=D, ok=r.get('correct', 0), n=r['midi'], rate=round(r.get('correct', 0)/max(r['midi'], 1), 4),
                         hands={h: [v[0], v[1]] for h, v in hand.items()}, cells={k: v for k, v in cells.items()})
        print(D, name, res[name]['ok'], '/', res[name]['n'], res[name]['rate'], flush=True)
    (RD.OUT/'screen_takes.json').write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
    lines = ['| 天 | 錄音 | 留一天正確率 | 左手 | 右手 | 最差的格子 |', '|---|---|---:|---:|---:|---|']
    for name, r in sorted(res.items(), key=lambda kv: kv[1]['rate']):
        hd = {h: f"{100*v[0]/v[1]:.0f}%" if v[1] else '' for h, v in r['hands'].items()}
        worst = sorted(((v[0]/v[1], k, v) for k, v in r['cells'].items() if v[1] >= 10))[:2]
        lines.append(f"| {r['day']} | {name} | {100*r['rate']:.1f}% | {hd.get('L', '')} | {hd.get('R', '')} | "
                     + '、'.join(f'{k} {v[0]}/{v[1]}' for _, k, v in worst) + ' |')
    (RD.OUT/'screen_takes.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('\n'.join(lines[:25]), flush=True)


if __name__ == '__main__':
    main()
