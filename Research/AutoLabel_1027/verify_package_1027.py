"""Replay the five pretests sample by sample through the INSTALLED 8th-package field engine
(automatic snare anchor 5) and score them; must agree with final_ens_1027.py (batch research code).
Also rewrites the reference replay CSVs used by Program/test_seventh_package.py for the R4-R7 tests,
checks heading invariance (+90 deg) on one recording, and replays today's pretest Raw."""
from __future__ import annotations
import csv, json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BM = ROOT/'Research/BestModel_20261021'
sys.path[:0] = [str(BM), str(HERE)]
import verify_1023 as V
import final_1027 as FB
L5, K = FB.L5, FB.K
core = V.core


def score(truth, P):
    pairs = V.match(truth, P); ok = 0; cells = {}
    for i, g in enumerate(truth):
        j = pairs.get(i); good = j is not None and (g['hand'] is None or P[j]['hand'] == g['hand']) and P[j]['drum'] == g['drum']
        ok += good
        if g['hand']:
            c = cells.setdefault(f"{g['hand']}|{g['drum']}", [0, 0, {}]); c[0] += good; c[1] += 1
            if not good:
                w = 'miss' if j is None else (P[j]['hand']+'手' if P[j]['hand'] != g['hand'] else P[j]['drum']); c[2][w] = c[2].get(w, 0)+1
    return ok, len(truth), cells


def main():
    FB.setup()  # registers today's pretest truth
    bundle, config = core.load_components()
    raws = {k: V.C['tests'][k]['raw'] for k in V.TESTS}; raws[L5.PRE] = str(L5.PRE_RAW)
    ref = BM/'out/replay_1023'; ref.mkdir(exist_ok=True); res = {}
    for D, names in FB.PRETESTS.items():
        ok = n = 0; cells = {}; used = []
        for nm in names:
            P, eng = V.stream(bundle, config, raws[nm], anchor=core.runtime.PRETEST_SNARE_ANCHOR)
            o, m, c = score(L5.TRUTH[nm], P); ok += o; n += m; used.append(dict(eng.runtime.anchored))
            for k, v in c.items():
                cc = cells.setdefault(k, [0, 0, {}]); cc[0] += v[0]; cc[1] += v[1]
                for w, q in v[2].items(): cc[2][w] = cc[2].get(w, 0)+q
            if nm in V.TESTS:
                with open(ref/f"{nm.replace(':', '_')}.csv", 'w', encoding='utf-8-sig', newline='') as f:
                    w = csv.DictWriter(f, fieldnames=['hand', 'peak_ms', 'emitted_ms', 'peak_g', 'drum', 'model_score']); w.writeheader()
                    for p in P: w.writerow({q: p[q] for q in w.fieldnames})
        res[D] = dict(ok=ok, n=n, rate=round(ok/n, 4), cells=cells, anchor_used=used)
        print('現場程式重播', D, ok, '/', n, round(ok/n, 4), used, {k: f'{v[0]}/{v[1]} {v[2]}' for k, v in sorted(cells.items()) if v[0] < v[1]}, flush=True)
    P0, _ = V.stream(bundle, config, V.C['tests']['seventh_15']['raw']); P1, _ = V.stream(bundle, config, V.C['tests']['seventh_15']['raw'], 90.0)
    inv = [(p['hand'], p['peak_ms'], p['drum']) for p in P0] == [(p['hand'], p['peak_ms'], p['drum']) for p in P1]
    print('heading +90 identical:', inv)
    res['heading_invariant'] = inv
    (HERE/'out/verify_package_1027.json').write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
