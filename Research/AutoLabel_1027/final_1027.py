"""10/27: score card + final model with rounds 4-8 (pretests never trained).

A  leave-one-day-out over five days: each day held out entirely (committee, drum means, hit gate);
   R8 held out = trained on R4-R7 (what the 10/27 field package was).
B  all five days trained (every non-pretest recording); the five pretests are scored (never trained,
   but the model has seen other recordings of the same day -> development score, not cross-day).
C  final bundle: committee 6 seeds + per-drum means + hit gate on all non-pretest candidates.
Pretests run like the field: snare anchor 5, per-drum offset correction, top-2 pose continuity.
"""
from __future__ import annotations
import json, sys
from collections import Counter
from pathlib import Path
import numpy as np
import joblib

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lodo5 as L5
K, H, LA, F = L5.K, L5.H, L5.LA, L5.F
import xgboost as xgb
sys.path.insert(0, str(L5.BM))
from final_model import portable_from

OUT = L5.OUT
DAYS = L5.DAYS5
PRETESTS = {'R4': ['pretest:round4'], 'R5': ['pretest:round5'], 'R6': ['pretest:round6'],
            'R7': ['seventh_01', 'seventh_12', 'seventh_15'], 'R8': [L5.PRE]}


def setup():
    docs = L5.r8_docs(); cf = OUT/'r8_cache.joblib'
    if cf.exists(): R8 = joblib.load(cf)
    else:
        from multiprocessing import Pool
        jobs = [(L5.PRE, str(L5.PRE_RAW), 1.5, {})]+[(n, d['raw_csv'], 1.5, {}) for n, d in docs.items()]
        with Pool(6) as pool: cand = dict(pool.imap_unordered(L5.hit_cand.work, jobs))
        R8 = dict(cand=cand, rows=L5.r8_rows(docs)); joblib.dump(R8, cf)
    H.CAND.update(R8['cand'])
    T8, truth = L5.r8_truth(docs, R8['cand']); L5.TRUTH[L5.PRE] = truth
    if L5.PRE not in K.TESTS: K.TESTS.append(L5.PRE)
    rows = joblib.load(K.OUT/'dataset2.joblib')['rows']+R8['rows']
    names = list(H.CAND); Lb = {n: L5.gate_labels(n, T8) for n in names}
    return rows, names, Lb, T8, R8


def pretest_score(D, com, tm, gate):
    cells, tot = {}, Counter()
    for n in PRETESTS[D]:
        P = LA.run(n, com, tm, gate, 5, 5); truth = L5.TRUTH[n]; pairs = L5.match(truth, P)
        for i, g in enumerate(truth):
            j = pairs.get(i); ok = j is not None and (g['hand'] is None or P[j]['hand'] == g['hand']) and P[j]['drum'] == g['drum']
            tot['ok'] += ok; tot['n'] += 1
            if g['hand']:
                c = cells.setdefault(f"{g['hand']}|{g['drum']}", [0, 0]); c[0] += ok; c[1] += 1
    return dict(ok=tot['ok'], n=tot['n'], rate=round(tot['ok']/tot['n'], 4), cells=cells)


def main():
    rows, names, Lb, T8, R8 = setup()
    print('rows', len(rows), 'R8', len(R8['rows']), flush=True)
    res = dict(lodo={}, full={})
    for D in DAYS:
        hold = {D}
        com, tm, gate, ntr = L5.train_day(rows, names, Lb, hold)
        r = L5.exam(D, names, com, tm, gate, T8); r['pretest_detail'] = pretest_score(D, com, tm, gate); r['train_rows'] = ntr
        res['lodo'][D] = r
        print('LODO', D, r['pretest_detail']['rate'], 'songs', r['songs'], flush=True)
        (OUT/'final_1027_scores.json').write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
    com, tm, gate, ntr = L5.train_day(rows, names, Lb, set())
    for D in DAYS:
        res['full'][D] = pretest_score(D, com, tm, gate); print('FULL', D, res['full'][D]['rate'], flush=True)
    res['full_train_rows'] = ntr
    (OUT/'final_1027_scores.json').write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
    # C: final bundle
    com6 = K.train(rows, {}, seeds=tuple(range(6)))
    old = joblib.load(L5.P8/'Model/venue_free_committee_1023.joblib')
    tmd = {f'{h}|{d}': np.mean([r['x'][83:104] for r in rows if r['hand'] == h and r['drum'] == d], 0).astype(np.float32)
           for h in 'LR' for d in K.DRUMS if any(r['hand'] == h and r['drum'] == d for r in rows)}
    gn = [n for n in names if n not in K.TESTS]
    Xg = np.vstack([np.stack([H.feats(c) for c in H.CAND[n]])[Lb[n][0] != 2] for n in gn]); yg = np.concatenate([Lb[n][0][Lb[n][0] != 2] for n in gn])
    m = xgb.XGBClassifier(n_estimators=800, max_depth=6, learning_rate=.05, subsample=.8, colsample_bytree=.6, device='cuda', tree_method='hist', random_state=0).fit(Xg, yg)
    bst = m.get_booster(); bst.set_param({'device': 'cpu'})
    bundle = dict(old, portable=portable_from(com6), members=[['all', s] for s in range(6)], classes=[str(c) for c in com6[0]['L'][0]],
                  training_events=len(rows), drum_rotation_means=tmd,
                  training='rounds 4-8: round4-6 reference labels, round7 songs (human + v2 auto labels), round7 protocol single-hand, '
                           'round8 (10/27) protocol single-hand + four-drum + song (v2/v3 auto labels, crosstalk excluded); pretests excluded',
                  hit_gate=dict(old['hit_gate'], xgb_json=bytes(bst.save_raw('json')), dims=int(Xg.shape[1]),
                                training=f'{len(gn)} non-pretest recordings (rounds 4-8), {len(yg)} candidates'))
    for k in [k for k in bundle if k.startswith('_')]: bundle.pop(k)
    joblib.dump(bundle, OUT/'venue_free_committee_rounds4to8_20261027.joblib', compress=3)
    print('saved final', len(rows), 'gate rows', len(yg), flush=True)


if __name__ == '__main__':
    main()
