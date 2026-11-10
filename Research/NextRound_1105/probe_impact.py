"""Can the hit's own dynamics (heading-free summary, no drum position) tell a cymbal from a tom?

Left hand, labelled training rows of every day (pretests are never in rows). Leave-one-day-out:
train on the other days, test on the held day. Feature sets:
  summary : x[:83]   (accel / gyro-free motion summary around the hit, no rotation-vs-zero block)
  rot     : x[83:]   (pose vs drumhead zero, 7 time offsets) - what separates drums by position
  all     : x
Tasks: 3-way {Ride, 中音 Tom, 落地 Tom} and binary Ride vs (中音+落地).
  HOLOGRIP_ROUNDS=out/rounds_r10pre.json python probe_impact.py
"""
import json, sys
import numpy as np
import xgboost as xgb
import rounds_data as RD

L5 = RD.L5
SETS = {'summary': slice(0, 83), 'rot': slice(83, None), 'all': slice(None)}
DR = ['Ride', '中音 Tom', '落地 Tom']


def main():
    rows, *_ = RD.setup()
    rs = [r for r in rows if r['hand'] == 'L' and r['drum'] in DR]
    X = np.stack([r['x'] for r in rs]).astype(np.float32); y3 = np.array([DR.index(r['drum']) for r in rs])
    day = np.array([L5.day_of(r['session']) for r in rs])
    days = sorted(set(day)); out = {}
    print('rows', len(rs), {d: int((day == d).sum()) for d in days}, 'dims', X.shape[1], flush=True)
    for task in ['ride_vs_tom', '3way']:
        y = (y3 == 0).astype(int) if task == 'ride_vs_tom' else y3
        for name, sl in SETS.items():
            res = {}
            for d in days:
                tr, te = day != d, day == d
                if len(set(y[te])) < 2: continue
                m = xgb.XGBClassifier(n_estimators=400, max_depth=6, learning_rate=.08, subsample=.8, colsample_bytree=.7,
                                      device='cuda', tree_method='hist', random_state=0).fit(X[tr][:, sl], y[tr])
                p = m.predict(X[te][:, sl])
                ride = y3[te] == 0
                res[d] = dict(acc=round(float((p == y[te]).mean()), 3),
                              ride_recall=round(float((p[ride] == (1 if task == 'ride_vs_tom' else 0)).mean()), 3) if ride.any() else None,
                              n=int(te.sum()), n_ride=int(ride.sum()))
            tot = sum(v['acc']*v['n'] for v in res.values())/sum(v['n'] for v in res.values())
            out[f'{task}|{name}'] = dict(mean_acc=round(tot, 3), days=res)
            print(task, name, round(tot, 3), {d: (v['acc'], v['ride_recall']) for d, v in res.items()}, flush=True)
    (RD.OUT/'probe_impact.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
