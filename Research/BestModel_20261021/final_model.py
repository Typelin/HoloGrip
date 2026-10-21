"""Train the selected configuration (base_ens: XGBoost+MLP per hand, sqrt class balance, x104) on all
13,617 rows with 6 seeds, export in the field-package portable format, and score it on the pretests
with the exact runtime probability code. Also scores the currently deployed committee for comparison."""
import json, sys
import numpy as np, joblib
import common as K
sys.path.insert(0, str(K.P6/'Program'))
from export_venue_free_1020 import export_mlp
import venue_free_runtime_1020 as rt


def portable_from(com):
    out = {}
    for s, member in enumerate(com):
        out[f'all_{s}'] = {}
        for h in 'LR':
            classes, models = member[h]
            bst = models[0].get_booster(); bst.set_param({'device': 'cpu'})
            out[f'all_{s}'][h] = dict(classes=classes, xgb_json=bytes(bst.save_raw('json')), mlp=export_mlp(models[1]))
    return out


def runtime_exam(bundle, D):
    rt.attach_boosters(bundle); res = {}
    for k, t in D['tests'].items():
        preds = []
        for h in 'LR':
            it = [c for c in t['items'] if c['hand'] == h]
            p = rt.probabilities(bundle, np.stack([c['x'] for c in it]), h)
            preds += [dict(t=c['t'], g=c['g'], drum=bundle['classes'][int(q.argmax())], score=float(q.max())) for c, q in zip(it, p)]
        res[k] = {th: K.score_pretest(t['truth'], preds, th) for th in (0.0, 0.6)}
    return res


def main():
    D = K.build(); rows = D['rows']
    com = K.train(rows, {}, seeds=tuple(range(6)))
    old = joblib.load(K.P7/'Model/venue_free_committee_1020.joblib')
    bundle = dict(old, portable=portable_from(com), members=[['all', s] for s in range(6)], classes=com[0]['L'][0],
                  training='rounds 4-7 (13,617 hits: round4-6 reference labels, round7 songs human+v2 labels, round7 protocol single-hand); pretests excluded',
                  pretests_in_training=False, compass_heading_used=False, per_hit_independent=True)
    assert com[0]['L'][0] == com[0]['R'][0]
    for name, b in [('new_rounds4to7', bundle), ('deployed_1020', dict(old))]:
        b['classes'] = [str(c) for c in b['classes']]
        res = runtime_exam(b, D)
        line = {k: f"{v[0.0]['correct']}/{v[0.0]['total']}={v[0.0]['correct']/v[0.0]['total']:.3f}" for k, v in res.items()}
        prec = sum(v[0.6]['correct'] for v in res.values())/sum(v[0.6]['out'] for v in res.values())
        mean = np.mean([v[0.0]['correct']/v[0.0]['total'] for v in res.values()])
        print(name, 'mean', round(float(mean), 4), 'precision@0.6', round(prec, 4), line, flush=True)
    for k in ('_boosters', '_xgb'): bundle.pop(k, None)
    joblib.dump(bundle, K.OUT/'venue_free_committee_rounds4to7_20261021.joblib', compress=3)
    print('saved')


if __name__ == '__main__':
    main()
