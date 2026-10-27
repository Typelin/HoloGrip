"""Leave-one-day-out with the automatic snare anchor: two ways to keep both clean drum positions
(re-centred training) and robustness to offsets (raw training).

mix  train on raw rows + re-centred copies of the same rows (data union); drum means from re-centred rows
ens  committee = raw-trained members + re-centred members (probabilities averaged); drum means from re-centred rows
Scored on pretests and songs/single-hand exactly like auto_anchor_1027.py.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import xgboost as xgb

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import auto_anchor_1027 as AA
RC, FB, L5, K, H, F = AA.RC, AA.FB, AA.L5, AA.K, AA.H, AA.F
ROT = RC.ROT


def gate_for(names, Lb, hold):
    gn = [n for n in names if n not in K.TESTS and L5.day_of(n) not in hold]
    Xg = np.vstack([np.stack([H.feats(c) for c in H.CAND[n]])[Lb[n][0] != 2] for n in gn]); yg = np.concatenate([Lb[n][0][Lb[n][0] != 2] for n in gn])
    return xgb.XGBClassifier(n_estimators=800, max_depth=6, learning_rate=.05, subsample=.8, colsample_bytree=.6, device='cuda', tree_method='hist', random_state=0).fit(Xg, yg)


def means(tr):
    return {h: np.array([np.mean([r['x'][ROT] for r in tr if r['hand'] == h and r['drum'] == d], 0) if any(r['hand'] == h and r['drum'] == d for r in tr) else np.zeros(21)
                         for d in K.DRUMS]) for h in 'LR'}


def score(D, names, com, tm, gate, T8):
    pre = dict(ok=0, n=0, cells={}); song = dict(ok=0, n=0, cells={}); st = {'pre': {'anchored': 0, 'not': 0}, 'song': {'anchored': 0, 'not': 0}}
    for nm in [x for x in names if L5.day_of(x) == D]:
        if nm in K.TESTS:
            P = AA.run(nm, com, tm, gate, st['pre']); truth = L5.TRUTH[nm]; pairs = L5.match(truth, P)
            for i, g in enumerate(truth):
                j = pairs.get(i); ok = j is not None and (g['hand'] is None or P[j]['hand'] == g['hand']) and P[j]['drum'] == g['drum']
                pre['ok'] += ok; pre['n'] += 1
                if g['hand']: c = pre['cells'].setdefault(f"{g['hand']}|{g['drum']}", [0, 0]); c[0] += ok; c[1] += 1
        else:
            used, excl = T8[nm] if nm in T8 else F.truth_of(nm)
            r, cl = F.score(used, excl, AA.run(nm, com, tm, gate, st['song']))
            song['ok'] += r.get('correct', 0); song['n'] += r['midi']
            for g, o in cl:
                if g['hand']: c = song['cells'].setdefault(f"{g['hand']}|{g['drum']}", [0, 0]); c[0] += o == 'ok'; c[1] += 1
    return dict(pretest=pre, songs=song, anchor_use=st)


def main():
    rows, names, Lb, T8, R8 = FB.setup()
    res = {}
    for D in FB.DAYS:
        tr = [r for r in rows if L5.day_of(r['session']) != D]; rc = RC.recenter(tr); tm = means(rc); gate = gate_for(names, Lb, {D})
        com_raw = K.train(tr, {}, seeds=(0, 1)); com_rc = K.train(rc, {}, seeds=(0, 1))
        com_mix = K.train(tr+rc, {}, seeds=(0, 1))
        for variant, com in (('mix', com_mix), ('ens', com_raw+com_rc)):
            r = score(D, names, com, tm, gate, T8); res[f'{D}|{variant}'] = r
            print(D, variant, '| 前測', r['pretest']['ok'], '/', r['pretest']['n'], round(r['pretest']['ok']/r['pretest']['n'], 4),
                  '| 歌曲單手', r['songs']['ok'], '/', r['songs']['n'], round(r['songs']['ok']/r['songs']['n'], 4), flush=True)
            (FB.OUT/'mix_1027.json').write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
