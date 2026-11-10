"""Which learner suits this data? Same 104 features, same rows, leave-one-day-out (labelled takes; pretests never in rows).

  knn      : standardised features, k nearest neighbours (distance-weighted)
  mlp      : the field MLP alone (512-256-128, GELU, BatchNorm, dropout; GPU)
  xgb      : the field XGBoost alone (GPU)
  xgb+mlp  : what the field committee members are (average of both)
Per-hand models, one seed, raw rows (the field package adds re-centred members, per-drum offset correction,
anchor and position prior on top; those help every learner and are not part of this comparison).
  HOLOGRIP_ROUNDS=out/rounds_r10pre.json python compare_learners.py
"""
import json, time
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import rounds_data as RD

L5, K = RD.L5, RD.FB.K
DR = list(K.DRUMS)


def fit_predict(kind, tr, te):
    out = {}
    for h in 'LR':
        a = [r for r in tr if r['hand'] == h]; b = [r for r in te if r['hand'] == h]
        if not b: continue
        Xb = np.stack([r['x'] for r in b])
        if kind == 'knn':
            Xa = np.stack([r['x'] for r in a]); mu, sd = Xa.mean(0), Xa.std(0)+1e-6
            m = KNeighborsClassifier(n_neighbors=15, weights='distance').fit((Xa-mu)/sd, [r['drum'] for r in a])
            out[h] = list(m.predict((Xb-mu)/sd))
        else:
            cfg = dict(xgb=kind != 'mlp', mlp=kind != 'xgb')
            classes, models = K.fit_hand(a, cfg, 0)
            p = np.mean([m.predict_proba(Xb) for m in models], 0)
            out[h] = [classes[i] for i in p.argmax(1)]
    return out


def main():
    rows, *_ = RD.setup()
    day = {id(r): L5.day_of(r['session']) for r in rows}
    days = sorted({*day.values()}, key=lambda d: int(d[1:]))
    res = {}
    for kind in ['knn', 'mlp', 'xgb', 'xgb+mlp']:
        t = time.time(); res[kind] = {}
        for d in days:
            tr = [r for r in rows if day[id(r)] != d]; te = [r for r in rows if day[id(r)] == d]
            pred = fit_predict(kind, tr, te)
            ok = n = 0; lr_ok = lr_n = 0
            for h, ps in pred.items():
                b = [r for r in te if r['hand'] == h]
                for r, p in zip(b, ps):
                    ok += p == r['drum']; n += 1
                    if h == 'L' and r['drum'] == 'Ride': lr_ok += p == 'Ride'; lr_n += 1
            res[kind][d] = dict(ok=int(ok), n=n, acc=round(ok/n, 4), left_ride=f'{lr_ok}/{lr_n}')
        tot_ok = sum(v['ok'] for v in res[kind].values()); tot_n = sum(v['n'] for v in res[kind].values())
        res[kind]['total'] = dict(ok=tot_ok, n=tot_n, acc=round(tot_ok/tot_n, 4), seconds=round(time.time()-t))
        print(kind, res[kind]['total'], {d: (v['acc'], v['left_ride']) for d, v in res[kind].items() if d != 'total'}, flush=True)
    (RD.OUT/'compare_learners.json').write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
