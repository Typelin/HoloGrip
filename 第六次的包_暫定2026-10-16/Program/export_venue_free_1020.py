"""Export the strict venue-free committee to a portable bundle and build the summary table.

Committee = A2+summary members with drumhead-zero jitter 10 and 15 deg, seeds 0-2 (6 members);
each member per hand = XGBoost (saved as JSON, CPU inference) + MLP (weights as numpy).
Pretests were never in training. Summary table: full pipeline, each hit judged alone
(own -240..+80 ms window + recorded drumhead zero), detector 2.1 g, ghost-cleaned answers.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path

import joblib
import numpy as np
from scipy.special import erf

PROG = Path(__file__).resolve().parent
sys.path.insert(0, str(PROG))
OUT = PROG.parent/'Data/鼓組對準模型_1020'
SEVENTH = Path(r'C:/Users/Typelin_Station/Desktop/HoloGrip/第七次的包_暫定2026-10-20')
MEMBERS = [(a, s) for a in (10, 15) for s in range(3)]


def export_mlp(m):
    layers = []
    mods = list(m.net)
    i = 0
    while i < len(mods):
        lin = mods[i]
        if i+1 < len(mods) and mods[i+1].__class__.__name__ == 'BatchNorm1d':
            bn = mods[i+1]
            layers.append(dict(W=lin.weight.detach().cpu().numpy().T.astype(np.float32), b=lin.bias.detach().cpu().numpy().astype(np.float32),
                               bn_mean=bn.running_mean.cpu().numpy(), bn_var=bn.running_var.cpu().numpy(), bn_w=bn.weight.detach().cpu().numpy(),
                               bn_b=bn.bias.detach().cpu().numpy(), bn_eps=bn.eps, act='gelu'))
            i += 4
        else:
            layers.append(dict(W=lin.weight.detach().cpu().numpy().T.astype(np.float32), b=lin.bias.detach().cpu().numpy().astype(np.float32), act=None))
            i += 1
    return dict(mu=m.mu.astype(np.float32), sd=m.sd.astype(np.float32), layers=layers, classes=list(m.classes_))


def mlp_proba(p, X):
    h = (X-p['mu'])/p['sd']
    for L in p['layers']:
        h = h@L['W']+L['b']
        if L['act']:
            h = (h-L['bn_mean'])/np.sqrt(L['bn_var']+L['bn_eps'])*L['bn_w']+L['bn_b']
            h = 0.5*h*(1.0+erf(h/np.sqrt(2.0)))
    h = h-h.max(1, keepdims=True); e = np.exp(h)
    return e/e.sum(1, keepdims=True)


def main():
    t0 = time.perf_counter()
    import gpu_strict_eval_1020 as G
    import xgboost as xgb
    sys.modules['__main__'].TorchMLP = G.TorchMLP
    com =joblib.load(OUT/'gpu_strict_committee.joblib')['members']
    portable = {}
    for key in MEMBERS:
        portable[key] = {}
        for h in 'LR':
            classes, ms = com[key][h]
            booster = ms['xgb'].get_booster(); booster.set_param({'device': 'cpu'})
            portable[key][h] = dict(classes=classes, xgb_json=bytes(booster.save_raw('json')), mlp=export_mlp(ms['mlp']))
    # consistency: portable vs original on pretest hits
    C = joblib.load(OUT/'pose_cache.joblib'); z = G.load()
    import pose_experiments_1020 as E
    def committee_proba(h, X, use_portable):
        out = 0
        for key in MEMBERS:
            if use_portable:
                pm = portable[key][h]; bst = xgb.Booster(); bst.load_model(bytearray(pm['xgb_json']))
                px = bst.predict(xgb.DMatrix(X), output_margin=False)
                out = out+(px+mlp_proba(pm['mlp'], X))/2
            else:
                out = out+G.proba(com[key][h][1], X)
        return out/len(MEMBERS)
    worst = 0.0
    for k in E.TESTS:
        for h in 'LR':
            items = [c for c in C['peaks'][k] if c['peak']['hand'] == h][:120]
            P = np.array([c['pose'] for c in items]); Z = np.array([list(z[k][h][:2])+[0.0]]*len(items), dtype=float)
            X = np.hstack([G.sub(np.array([c['hf'] for c in items], dtype=np.float32), 'summary'), G.rot_batch(P, Z)]).astype(np.float32)
            worst = max(worst, float(np.abs(committee_proba(h, X, True)-committee_proba(h, X, False)).max()))
    print('portable vs original max prob diff', worst, flush=True)
    assert worst < 1e-3
    det = dict(C['det'])
    bundle = dict(contract='hologrip_venue_free_committee_1020_v1', members=[list(k) for k in MEMBERS], portable={f'{k[0]}_{k[1]}': v for k, v in portable.items()},
                  detector_config=det, classes=portable[MEMBERS[0]]['L']['classes'], feature='heading_free summary(83) + rot vs drumhead zero (7x3)',
                  pose_dts_ms=[-120, -80, -40, -20, 0, 20, 40], compass_heading_used=False, pretests_in_training=False,
                  future_context_ms=80, per_hit_independent=True)
    joblib.dump(bundle, OUT/'venue_free_committee_1020.joblib', compress=3)
    # summary table: ten recordings + pretests (full pipeline, per-hit)
    table = {}
    for s in E.SESS:
        name = s['name']; items = sorted(C['peaks'][name], key=lambda c: c['peak']['peak_ms'])
        preds = []
        for h in 'LR':
            it = [c for c in items if c['peak']['hand'] == h]
            if not it or z[name][h] is None: continue
            P = np.array([c['pose'] for c in it]); Z = np.array([list(z[name][h][:2])+[0.0]]*len(it), dtype=float)
            X = np.hstack([G.sub(np.array([c['hf'] for c in it], dtype=np.float32), 'summary'), G.rot_batch(P, Z)]).astype(np.float32)
            d = np.array(bundle['classes'])[committee_proba(h, X, True).argmax(1)]
            preds += [dict(**{q: v for q, v in c['peak'].items() if q != 'feature'}, drum=str(x), model_score=1.0) for c, x in zip(it, d)]
        preds.sort(key=lambda p: p['peak_ms'])
        g = set(E.GHOST[name]['ghosts'])
        b, _, _ = E.H.first.training.assess(dict(s, truth=[x for x in s['truth'] if x['id'] not in g]), preds)
        table[name] = dict(correct=b['correct'], total=b['total'], rate=round(b['rate'], 4), missed=b.get('missed'), wrong=b.get('wrong_drum'), extra=b.get('unmatched_extra'))
        print(name, table[name], flush=True)
    for k, t in E.TESTS.items():
        g = set(E.GHOST[k]['ghosts']); S = dict(t['session'], truth=[x for x in t['session']['truth'] if x['id'] not in g])
        tt = np.array([float(x['t']) for x in S['truth']]); lo, hi = tt.min()-200, tt.max()+200
        preds = []
        for h in 'LR':
            it = [c for c in C['peaks'][k] if c['peak']['hand'] == h and lo <= c['peak']['peak_ms'] <= hi]
            if not it: continue
            P = np.array([c['pose'] for c in it]); Z = np.array([list(z[k][h][:2])+[0.0]]*len(it), dtype=float)
            X = np.hstack([G.sub(np.array([c['hf'] for c in it], dtype=np.float32), 'summary'), G.rot_batch(P, Z)]).astype(np.float32)
            d = np.array(bundle['classes'])[committee_proba(h, X, True).argmax(1)]
            preds += [dict(**{q: v for q, v in c['peak'].items() if q != 'feature'}, drum=str(x), model_score=1.0) for c, x in zip(it, d)]
        preds.sort(key=lambda p: p['peak_ms'])
        sc = E.H.first.scoring.score(S, preds)[0]
        table[k] = dict(correct=sc['correct'], total=sc['total'], rate=round(sc['correct']/sc['total'], 4), missed=sc['missed'], wrong=sc['wrong_drum'], extra=sc.get('extra_unmatched_in_span'))
        print(k, table[k], flush=True)
    (OUT/'venue_free_committee_總表.json').write_text(json.dumps(table, ensure_ascii=False, indent=1), encoding='utf-8')
    print('done', round(time.perf_counter()-t0), 's')


if __name__ == '__main__':
    main()
