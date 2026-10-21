"""Does weighting training hits by peak strength improve accuracy?

Training: round 4/5/6 reference labels + today's conservative auto (03-11) + finished song 13.
Held out: all pretests (round 4/5/6, today 01/12/15) and song 14. Pretests never trained.
peak_g comes from feature x[17] = log1p(max |acc| in the hit window).
Weights (multiplied with the usual 1/sqrt(class count) balance):
  W0 uniform | W1 linear 0.3..1 over 2.1..3.0 g | W2 drop hits < 2.6 g | W3 strong 0.1..1 over 2.1..3.0 g
"""
import json
from collections import Counter
from pathlib import Path
import numpy as np
import soft_hit_experiment as S

G = S.G


def peak_g(x): return float(np.expm1(x[17]))


def weight(g, mode):
    if mode == 'W0': return 1.0
    if mode == 'W1': return float(np.clip(0.3+0.7*(g-2.1)/0.9, 0.3, 1.0))
    if mode == 'W2': return 1.0 if g >= 2.6 else 0.0
    if mode == 'W3': return float(np.clip(0.1+0.9*(g-2.1)/0.9, 0.1, 1.0))


def train(rows, mode, seed):
    models = {}
    for h in 'LR':
        r = [q for q in rows if q['hand'] == h and weight(q['g'], mode) > 0]
        X = np.stack([q['x'] for q in r]); y = [q['drum'] for q in r]
        c = Counter(y); w = np.array([weight(q['g'], mode)/np.sqrt(c[q['drum']]) for q in r], np.float32); w /= w.mean()
        models[h] = G.fit_models(X, y, w, 'ens', seed=seed)
    return models


def main():
    lab = json.loads(S.LAB13.read_text(encoding='utf-8'))['labels']
    r13, _ = S.rows_from_labels('seventh_13', lab)
    rows = [dict(hand=r['hand'], drum=r['drum'], x=np.asarray(r['x'], np.float32)) for r in S.C['baseline']]
    rows += [dict(hand=r['hand'], drum=r['drum'], x=np.asarray(r['x'], np.float32)) for r in S.C['auto']
             if r['session'].startswith('seventh_') and r['session'] not in ('seventh_13', 'seventh_14')]
    rows += [dict(hand=r['hand'], drum=r['drum'], x=r['x']) for r in r13]
    for q in rows: q['g'] = peak_g(q['x'])
    gs = np.array([q['g'] for q in rows]); print('training rows', len(rows), 'peak g quartiles', np.percentile(gs, [5, 25, 50, 75]).round(2), 'share<2.6', round(float((gs < 2.6).mean()), 3), flush=True)
    T = S.tests(); res = {}
    for mode in ['W0', 'W1', 'W2', 'W3']:
        ms = [train(rows, mode, s) for s in range(2)]; res[mode] = {}
        for k, (truth, items) in T.items():
            preds = S.predict(ms, items)
            res[mode][k] = {f'thr{t}': S.score(truth, preds, t) for t in (0.0, 0.6)}
        line = {k: f"{v['thr0.0']['rate']:.3f}/{v['thr0.6']['precision']:.3f}" for k, v in res[mode].items()}
        print(mode, line, flush=True)
    (Path(__file__).parent/'out/weight_experiment.json').write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
