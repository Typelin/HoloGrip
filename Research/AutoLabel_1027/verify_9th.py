"""Replay the five pretests sample by sample through the 9th package field engine and score them
(answer key without electronic-drum double triggers, truth_clean_1027.py).
Writes reference replays for Program/test_seventh_package.py (out/replay_9th) and checks heading invariance."""
from __future__ import annotations
import csv, json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
P9 = ROOT/'第九次的包_暫定'
sys.path[:0] = [str(P9/'Program'), str(HERE)]
import seventh_core as core
assert Path(core.__file__).resolve().parent == (P9/'Program').resolve()
import final_1027 as FB
import verify_package_1027 as VP
import truth_clean_1027 as TC
L5, V = FB.L5, VP.V


def stream(bundle, config, raw, turn=0.0, anchor=0):
    hands = core.trajectory.read_raw(raw)
    with open(raw, encoding='utf-8-sig', newline='') as f: source = list(csv.DictReader(f))
    engine = core.Engine(bundle, config, snare_anchor=anchor); out = []
    for _, h, s in sorted((s['row_index'], h, s) for h, a in hands.items() for s in a):
        row = source[s['row_index']]; s = dict(s, absolute_yaw=(s['absolute_yaw']+turn+180) % 360-180)
        frame = dict(ax=s['ax'], ay=s['ay'], az=s['az'], yaw=(float(row['yaw_deg'])+turn+180) % 360-180, pitch=float(row['pitch_deg']), roll=s['roll'])
        p, reason = engine.push(h, s, frame); assert reason == '', reason; out.extend(p)
    return sorted(out, key=lambda p: (p['peak_ms'], p['hand'])), engine


def main():
    FB.setup(); bundle, config = core.load_components()
    raws = {k: V.C['tests'][k]['raw'] for k in V.TESTS}; raws[L5.PRE] = str(L5.PRE_RAW)
    ref = HERE/'out/replay_9th'; ref.mkdir(exist_ok=True); res = {}; tot = [0, 0]
    for D, names in FB.PRETESTS.items():
        ok = n = 0; cells = {}; used = []
        for nm in names:
            P, eng = stream(bundle, config, raws[nm], anchor=core.runtime.PRETEST_SNARE_ANCHOR)
            o, m, c = VP.score(TC.dedupe(L5.TRUTH[nm], FB.H.CAND[nm])[0], P); ok += o; n += m; used.append(dict(eng.runtime.anchored))
            for k, v in c.items():
                cc = cells.setdefault(k, [0, 0, {}]); cc[0] += v[0]; cc[1] += v[1]
                for w, q in v[2].items(): cc[2][w] = cc[2].get(w, 0)+q
            with open(ref/f"{nm.replace(':', '_')}.csv", 'w', encoding='utf-8-sig', newline='') as f:
                w = csv.DictWriter(f, fieldnames=['hand', 'peak_ms', 'emitted_ms', 'peak_g', 'drum', 'model_score']); w.writeheader()
                for p in P: w.writerow({q: p[q] for q in w.fieldnames})
        tot[0] += ok; tot[1] += n
        res[D] = dict(ok=ok, n=n, rate=round(ok/n, 4), cells=cells, anchor_used=used)
        print('第九次包重播', D, ok, '/', n, round(ok/n, 4), used, {k: f'{v[0]}/{v[1]} {v[2]}' for k, v in sorted(cells.items()) if v[0]/v[1] < .85}, flush=True)
    print('合計', tot, round(tot[0]/tot[1], 4))
    P0, _ = stream(bundle, config, V.C['tests']['seventh_15']['raw']); P1, _ = stream(bundle, config, V.C['tests']['seventh_15']['raw'], 90.0)
    inv = [(p['hand'], p['peak_ms'], p['drum']) for p in P0] == [(p['hand'], p['peak_ms'], p['drum']) for p in P1]
    print('heading +90 identical:', inv); res['heading_invariant'] = inv; res['total'] = tot
    (HERE/'out/verify_9th.json').write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
