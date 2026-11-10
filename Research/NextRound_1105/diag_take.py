"""Trace one labelled take through the field engine hit by hit: anchor decision, zero-direction guard,
pose at the hit (zero-relative yaw/pitch from the Raw CSV) and the answer from the labels.

  python diag_take.py --code 第十次的包_暫定 --take "R9:ton1轉ton2 L" [--anchor 0] [--hand L]
"""
from __future__ import annotations
import argparse, bisect, csv, json, sys
from collections import Counter
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--code', default='第十次的包_暫定')
ap.add_argument('--day', default='R9')
ap.add_argument('--take', required=True)
ap.add_argument('--anchor', type=int, default=-1, help='snare anchor hits (-1 = field default 5, 0 = off)')
ap.add_argument('--hand', default='')
ap.add_argument('--quiet', action='store_true')
ap.add_argument('--no-cont', action='store_true', help='diagnostic only: pose continuity off')
ap.add_argument('--no-adapt', action='store_true', help='diagnostic only: per-drum offset correction off')
ap.add_argument('--no-prior', action='store_true', help='diagnostic only: position prior off')
ARGS = ap.parse_args() if __name__ == '__main__' else ap.parse_args(['--take', '_'])

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT/ARGS.code/'Program'), str(HERE)]
import seventh_core as core
import rounds_data as RD
import verify_package_1027 as VP


def main():
    RD.setup(); bundle, config = core.load_components()
    if ARGS.no_cont: core.runtime.CONTINUITY['max_ms'] = -1.0
    if ARGS.no_adapt: bundle['adaptation'] = dict(bundle['adaptation'], min_confident_hits=10**9)
    if ARGS.no_prior: bundle['position_prior'] = dict(bundle['position_prior'], weight=0.0)
    d = RD.docs_of(ARGS.day, RD.config()[ARGS.day])[ARGS.take]; raw = d['raw_csv']
    used = [dict(t=float(e['csv_ms']), drum=d['labels'][str(e['id'])]['drum'], hand=d['labels'][str(e['id'])]['mark'])
            for e in d['events'] if d['labels'][str(e['id'])]['mark'] in ('L', 'R')]
    hands = core.trajectory.read_raw(raw)
    with open(raw, encoding='utf-8-sig', newline='') as f: source = list(csv.DictReader(f))
    anchor = core.runtime.PRETEST_SNARE_ANCHOR if ARGS.anchor < 0 else ARGS.anchor
    eng = core.Engine(bundle, config, snare_anchor=anchor); P = []
    for _, h, s in sorted((s['row_index'], h, s) for h, a in hands.items() for s in a):
        row = source[s['row_index']]
        p, reason = eng.push(h, s, dict(ax=s['ax'], ay=s['ay'], az=s['az'], yaw=float(row['yaw_deg']), pitch=float(row['pitch_deg']), roll=s['roll']))
        P.extend(p)
    P.sort(key=lambda p: (p['peak_ms'], p['hand']))
    ok, n, cells = VP.score(used, P)
    rt = eng.runtime
    print(json.dumps(dict(take=ARGS.take, code=ARGS.code, anchor_hits=anchor, off=[k for k in ('no_cont', 'no_adapt', 'no_prior') if getattr(ARGS, k)], ok=ok, n=n, rate=round(ok/n, 4),
                          anchored={h: rt.anchored[h] for h in 'LR'},
                          anchored_drum={h: (rt.classes[rt.anchored_drum[h]] if rt.anchored_drum[h] is not None else None) for h in 'LR'},
                          zero_rot={h: float(rt.zero_rot[h]) for h in 'LR'},
                          cells={k: f'{v[0]}/{v[1]} {v[2]}' for k, v in sorted(cells.items())}), ensure_ascii=False))
    if ARGS.quiet: return
    rows = {h: [r for r in source if r['hand'] == h] for h in 'LR'}; ts = {h: [float(r['song_time_ms']) for r in rows[h]] for h in 'LR'}
    ut = sorted(used, key=lambda g: g['t'])
    for p in P:
        if ARGS.hand and p['hand'] != ARGS.hand: continue
        h = p['hand']; t = float(p['peak_ms']); i = min(bisect.bisect_left(ts[h], t), len(ts[h])-1); r = rows[h][i]
        g = min(ut, key=lambda g: abs(g['t']-t)); ans = g['drum'] if abs(g['t']-t) <= 80 else '-'
        print(f"{h} {t/1000:7.2f}s pred={p['drum']:<7s} ans={ans:<7s} score={float(p['model_score']):.2f} yaw={float(r['cal_yaw_deg']):6.1f} pitch={float(r['cal_pitch_deg']):6.1f} zero={p.get('zero_correction_deg', 0):.0f}")


if __name__ == '__main__':
    main()
