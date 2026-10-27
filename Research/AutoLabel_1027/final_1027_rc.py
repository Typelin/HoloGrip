"""Same score card as final_1027.py, but every training recording is first re-centred per hand.

Finding: with all five days trained, the 10/27 pretest left hand was still Ride 0/20 and 中音 4/20,
although 10/27 single-hand Ride/中音 takes were in training. Each take has its own drumhead zero
(left-hand zero yaw 95-148 deg across today's takes), so the same drum lands at different
zero-relative rotations in different takes. At runtime the per-drum offset correction (snare anchor)
moves a take onto the training drum means; the training rows themselves were never moved.

Re-centring (training side only, labels of that recording only): for each (recording, hand) shift
the rotation block x[83:104] by the count-weighted mean of (global drum mean - take drum mean), two
passes. Global means come from the training rows only (held-out day never used).
"""
from __future__ import annotations
import json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import joblib

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import final_1027 as FB
L5, K, H = FB.L5, FB.K, FB.H
import xgboost as xgb
from final_model import portable_from

ROT = slice(83, 104)


def recenter(rows, passes=2):
    rows = [dict(r, x=r['x'].copy()) for r in rows]
    for _ in range(passes):
        G = {}
        for h in 'LR':
            for d in K.DRUMS:
                v = [r['x'][ROT] for r in rows if r['hand'] == h and r['drum'] == d]
                if v: G[(h, d)] = np.mean(v, 0)
        grp = defaultdict(list)
        for r in rows: grp[(r['session'], r['hand'])].append(r)
        for (s, h), rs in grp.items():
            by = defaultdict(list)
            for r in rs: by[r['drum']].append(r['x'][ROT])
            delta = sum(len(v)*(G[(h, d)]-np.mean(v, 0)) for d, v in by.items())/len(rs)
            for r in rs: r['x'][ROT] += delta
    return rows


def train_day_rc(rows, names, Lb, hold):
    tr = recenter([r for r in rows if L5.day_of(r['session']) not in hold])
    com = K.train(tr, {}, seeds=(0, 1))
    tm = {h: np.array([np.mean([r['x'][ROT] for r in tr if r['hand'] == h and r['drum'] == d], 0) if any(r['hand'] == h and r['drum'] == d for r in tr) else np.zeros(21)
                       for d in K.DRUMS]) for h in 'LR'}
    gn = [n for n in names if n not in K.TESTS and L5.day_of(n) not in hold]
    Xg = np.vstack([np.stack([H.feats(c) for c in H.CAND[n]])[Lb[n][0] != 2] for n in gn]); yg = np.concatenate([Lb[n][0][Lb[n][0] != 2] for n in gn])
    gate = xgb.XGBClassifier(n_estimators=800, max_depth=6, learning_rate=.05, subsample=.8, colsample_bytree=.6, device='cuda', tree_method='hist', random_state=0).fit(Xg, yg)
    return com, tm, gate, len(tr), tr


def main():
    rows, names, Lb, T8, R8 = FB.setup()
    res = dict(lodo={}, full={})
    for D in FB.DAYS:
        com, tm, gate, ntr, _ = train_day_rc(rows, names, Lb, {D})
        r = L5.exam(D, names, com, tm, gate, T8); r['pretest_detail'] = FB.pretest_score(D, com, tm, gate); r['train_rows'] = ntr
        res['lodo'][D] = r; print('RC LODO', D, r['pretest_detail']['rate'], 'songs', r['songs'], flush=True)
        (FB.OUT/'final_1027_rc_scores.json').write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
    com, tm, gate, ntr, tr = train_day_rc(rows, names, Lb, set())
    for D in FB.DAYS:
        res['full'][D] = FB.pretest_score(D, com, tm, gate); print('RC FULL', D, res['full'][D]['rate'], flush=True)
    (FB.OUT/'final_1027_rc_scores.json').write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
    if '--save' in sys.argv:
        com6 = K.train(tr, {}, seeds=tuple(range(6)))
        old = joblib.load(L5.P8/'Model/venue_free_committee_1023.joblib')
        tmd = {f'{h}|{d}': np.mean([r['x'][ROT] for r in tr if r['hand'] == h and r['drum'] == d], 0).astype(np.float32)
               for h in 'LR' for d in K.DRUMS if any(r['hand'] == h and r['drum'] == d for r in tr)}
        b = joblib.load(FB.OUT/'venue_free_committee_rounds4to8_20261027.joblib')
        b.update(portable=portable_from(com6), members=[['all', s] for s in range(6)], classes=[str(c) for c in com6[0]['L'][0]], drum_rotation_means=tmd,
                 training=b['training']+'; training rotation block re-centred per recording and hand (labels of that recording only)')
        joblib.dump(b, FB.OUT/'venue_free_committee_rounds4to8_rc_20261027.joblib', compress=3); print('saved rc final', flush=True)


if __name__ == '__main__':
    main()
