"""Build a field model bundle (same recipe as the 9th package) from rounds 4-8 plus the enabled new days.

Recipe (identical to the 9th package, Research/AutoLabel_1027 final_1027 / final_ens_1027 + the
weighting, position prior, anchor and zero-direction guard added for the 9th package):
  committee      3 seeds on raw rows + 3 seeds on rows re-centred per recording and hand
                 (member weights 0.4 / 0.6 split evenly), drum means from the re-centred rows
  hit gate       XGBoost on all non-pretest candidates (126 dims, threshold 0.4)
  zero estimator heading-free XGBoost per hand (115 dims) for the zero-direction guard
  everything else (adaptation, position prior, anchor rule, guard thresholds, detector) is copied
  from the 9th package model unchanged.
Pretests are never trained. All training on GPU.

  python build_model.py --tag tenth                 # all days -> out/model_tenth.joblib
  python build_model.py --tag lodo_R9 --hold R9     # leave one day out (1 raw + 1 re-centred seed)
  python build_model.py --tag rebuild9 --days none  # rounds 4-8 only (rebuild check of the 9th model)
"""
from __future__ import annotations
import argparse, json, sys, time
from collections import Counter
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--tag', required=True)
ap.add_argument('--hold', default='', help='comma separated days left out of training (leave-one-day-out)')
ap.add_argument('--days', default='all', help="new days to add from rounds.json: 'all', 'none' or comma list")
ap.add_argument('--seeds', type=int, default=0, help='seeds per half (default 3 full / 1 with --hold)')
ARGS = ap.parse_args()   # parsed before the research imports (lodo_full rewrites sys.argv)

import numpy as np
import joblib
import xgboost as xgb

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rounds_data as RD
FB, L5, K, H = RD.FB, RD.L5, RD.K, RD.H
import final_1027_rc as RC
import yawfree_1027 as YF
from final_model import portable_from

TEMPLATE = RD.ROOT/'第九次的包_暫定/Model/venue_free_committee_1023.joblib'
ROT = RC.ROT


def cpu_json(model):
    b = model.get_booster(); b.set_param({'device': 'cpu'}); return bytes(b.save_raw('json'))


def main():
    t0 = time.time()
    extra = None if ARGS.days == 'all' else ([] if ARGS.days == 'none' else ARGS.days.split(','))
    rows, names, Lb, T, info = RD.setup(extra)
    hold = {d for d in ARGS.hold.split(',') if d}
    tr = [r for r in rows if L5.day_of(r['session']) not in hold]
    ns = ARGS.seeds or (1 if hold else 3); seeds = tuple(range(ns))
    print('training rows', len(tr), 'of', len(rows), 'hold', sorted(hold), 'seeds', ns, flush=True)
    rc = RC.recenter(tr)
    com = K.train(tr, {}, seeds=seeds)+K.train(rc, {}, seeds=seeds)
    tmd = {f'{h}|{d}': np.mean([r['x'][ROT] for r in rc if r['hand'] == h and r['drum'] == d], 0).astype(np.float32)
           for h in 'LR' for d in K.DRUMS if any(r['hand'] == h and r['drum'] == d for r in rc)}
    missing = [f'{h}|{d}' for h in 'LR' for d in K.DRUMS if f'{h}|{d}' not in tmd]
    assert not missing, f'no training rows for {missing}'
    gn = [n for n in names if n not in K.TESTS and L5.day_of(n) not in hold]
    Xg = np.vstack([np.stack([H.feats(c) for c in H.CAND[n]])[Lb[n][0] != 2] for n in gn]); yg = np.concatenate([Lb[n][0][Lb[n][0] != 2] for n in gn])
    gate = xgb.XGBClassifier(n_estimators=800, max_depth=6, learning_rate=.05, subsample=.8, colsample_bytree=.6, device='cuda', tree_method='hist', random_state=0).fit(Xg, yg)
    zero = YF.train(tr)
    days = sorted({L5.day_of(r['session']) for r in tr})
    b = joblib.load(TEMPLATE)
    keys = [f'all_{i}' for i in range(2*ns)]
    b.update(portable=portable_from(com), members=[['all', i] for i in range(2*ns)], classes=[str(c) for c in com[0]['L'][0]],
             member_weights={k: (0.4 if i < ns else 0.6)/ns for i, k in enumerate(keys)},
             member_groups=dict(raw=keys[:ns], recentred=keys[ns:]),
             drum_rotation_means=tmd, training_events=len(tr),
             training=f"days {','.join(days)} (pretests excluded): labelled songs, four-drum and protocol single-hand takes. "
                      f'Committee = {ns} seeds on raw rows + {ns} seeds on rows re-centred per recording and hand '
                      '(rotation block shifted onto the global drum means using that recording\'s own labels)',
             hit_gate=dict(b['hit_gate'], xgb_json=cpu_json(gate), dims=int(Xg.shape[1]),
                           training=f'{len(gn)} non-pretest recordings ({",".join(days)}), {len(yg)} candidates'),
             zero_estimator=dict(b['zero_estimator'], models={h: cpu_json(zero[h]) for h in 'LR'},
                                 training=f'days {",".join(days)} non-pretest rows ({len(tr):,}), XGBoost per hand'))
    assert b['classes'] == ['Crash', 'Hi-Hat', 'Ride', '中音 Tom', '小鼓', '落地 Tom', '高音 Tom'], b['classes']
    assert not b['pretests_in_training'] and not b['compass_heading_used']
    trained = {r['session'] for r in tr}
    assert not trained & set(K.TESTS), 'a pretest reached training'
    for k in [k for k in b if k.startswith('_')]: b.pop(k)
    out = RD.OUT/f'model_{ARGS.tag}.joblib'; joblib.dump(b, out, compress=3)
    summary = dict(tag=ARGS.tag, model=str(out), built=time.strftime('%Y-%m-%d'), hold=sorted(hold), days=days, seeds_per_half=ns, training_rows=len(tr),
                   rows_by_day=dict(Counter(L5.day_of(r['session']) for r in tr)), gate_candidates=int(len(yg)), gate_recordings=len(gn),
                   pretests=sorted(K.TESTS), new_days=info, minutes=round((time.time()-t0)/60, 1))
    (RD.OUT/f'model_{ARGS.tag}.json').write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
