"""Final 10/27 model: raw-trained + re-centred committee (ensemble), rounds 4-8, automatic snare anchor.

Chosen from leave-one-day-out with the automatic anchor (auto_anchor_1027.py, mix_1027.py):
  pretest  raw 79.2%  re-centred 82.7%  mix 80.4%  ensemble 83.9%
  songs    raw 77.9%  re-centred 74.0%  mix 76.7%  ensemble 75.6%
Also scores the five pretests with everything trained (development score; pretests never trained).
Members: 3 raw seeds + 3 re-centred seeds; drum means from the re-centred rows; hit gate on all
non-pretest candidates of rounds 4-8.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import joblib

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mix_1027 as MX
AA, RC, FB, L5, K = MX.AA, MX.RC, MX.FB, MX.L5, MX.K
from final_model import portable_from


def main():
    rows, names, Lb, T8, R8 = FB.setup()
    rc = RC.recenter(rows); tm = MX.means(rc); gate = MX.gate_for(names, Lb, set())
    com = K.train(rows, {}, seeds=(0, 1, 2))+K.train(rc, {}, seeds=(0, 1, 2))
    res = {}
    for D in FB.DAYS:
        st = {'anchored': 0, 'not': 0}; cells = {}; ok = n = 0
        for nm in FB.PRETESTS[D]:
            P = AA.run(nm, com, tm, gate, st); truth = L5.TRUTH[nm]; pairs = L5.match(truth, P)
            for i, g in enumerate(truth):
                j = pairs.get(i); good = j is not None and (g['hand'] is None or P[j]['hand'] == g['hand']) and P[j]['drum'] == g['drum']
                ok += good; n += 1
                if g['hand']: c = cells.setdefault(f"{g['hand']}|{g['drum']}", [0, 0]); c[0] += good; c[1] += 1
        res[D] = dict(ok=ok, n=n, rate=round(ok/n, 4), cells=cells, anchor_use=st)
        print('FULL ens', D, ok, '/', n, round(ok/n, 4), st, flush=True)
    (FB.OUT/'final_ens_full_scores.json').write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
    b = joblib.load(FB.OUT/'venue_free_committee_rounds4to8_20261027.joblib')
    tmd = {f'{h}|{d}': tm[h][i].astype(np.float32) for h in 'LR' for i, d in enumerate(K.DRUMS) if any(r['hand'] == h and r['drum'] == d for r in rc)}
    b.update(portable=portable_from(com), members=[['all', s] for s in range(len(com))], classes=[str(c) for c in com[0]['L'][0]],
             drum_rotation_means=tmd, training_events=len(rows),
             training='rounds 4-8 (pretests excluded): round4-6 reference labels, round7 songs (human + v2 auto) and protocol single-hand, '
                      'round8 (10/27) protocol single-hand + four-drum + song (v2/v3 auto, crosstalk excluded). Committee = 3 seeds trained on raw rows '
                      '+ 3 seeds trained on rows re-centred per recording and hand (rotation block shifted onto the global drum means using that recording\'s own labels)',
             anchor_rule=dict(kind='automatic_snare_anchor', hits=5, snare_in_top=2, max_pose_deg=8.0))
    joblib.dump(b, FB.OUT/'venue_free_committee_rounds4to8_ens_20261027.joblib', compress=3)
    print('saved ens final', len(rows), len(com), flush=True)


if __name__ == '__main__':
    main()
