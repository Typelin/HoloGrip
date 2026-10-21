"""Validation of kit-map calibration on SONGS of a day the model never saw.
Model: rounds 4-6 only. Calibration: first 3 hits of every (hand, drum) block of a round-7 pretest
(01 for the original kit -> songs 03, 04, 05; 15 for the moved kit -> songs 13, 14).
Score: labelled song hits (hand known), with and without calibration."""
import numpy as np, joblib
from collections import defaultdict
from scipy.optimize import linear_sum_assignment
import common as K, cells as Q

ROT = slice(83, 104)


def calib_centres(D2, T, k, Kc=3):
    truth = T[k]; items = D2['tests'][k]['items']; tt = np.array([g['t'] for g in truth]); pt = np.array([c['t'] for c in items])
    dt = np.abs(tt[:, None]-pt[None, :]); hm = np.array([[g['hand'] is not None and g['hand'] != c['hand'] for c in items] for g in truth])
    cost = np.hstack([np.where(dt <= 80, dt+300*hm, 1e9), np.full((len(tt), len(tt)), 1e6)])
    r, c = linear_sum_assignment(cost); pairs = {i: j for i, j in zip(r, c) if j < len(pt) and cost[i, j] <= 380}
    seen = defaultdict(int); cal = defaultdict(list)
    for i in sorted(range(len(truth)), key=lambda i: truth[i]['t']):
        g = truth[i]
        if g['hand'] and seen[(g['hand'], g['drum'])] < Kc and i in pairs:
            seen[(g['hand'], g['drum'])] += 1; cal[(g['hand'], g['drum'])].append(items[pairs[i]]['x'][ROT])
    return {key: np.mean(v, 0) for key, v in cal.items()}


def classify(com, h, X, cen, w, sig=.15):
    p = K.proba(com, h, X)
    if w == 0: return p
    dist = np.full((len(X), 7), 9.)
    for d_i, d in enumerate(K.DRUMS):
        if (h, d) in cen: dist[:, d_i] = np.linalg.norm(X[:, ROT]-cen[(h, d)], axis=1)
    q = np.exp(-dist**2/sig**2); q /= q.sum(1, keepdims=True)+1e-12
    return p**(1-w)*(q+1e-6)**w


def main():
    D2 = joblib.load(K.OUT/'dataset2.joblib'); T = Q.truth_hands()
    com = K.train([r for r in D2['rows'] if r['group'] in ('R4', 'R5', 'R6')], {}, seeds=(0, 1))
    plan = {'seventh_01': ['seventh_03', 'seventh_04', 'seventh_05'], 'seventh_15': ['seventh_13', 'seventh_14'], 'seventh_12': ['seventh_13', 'seventh_14']}
    for pk, songs in plan.items():
        cen = calib_centres(D2, T, pk)
        for w in (0, .5, .7):
            line = []
            for s in songs:
                rs = [r for r in D2['rows'] if r['session'] == s]; ok = n = 0; cell = defaultdict(lambda: [0, 0])
                for h in 'LR':
                    t = [r for r in rs if r['hand'] == h]
                    if not t: continue
                    p = classify(com, h, np.stack([r['x'] for r in t]), cen, w)
                    for r, q in zip(t, p):
                        okk = K.DRUMS[int(q.argmax())] == r['drum']; ok += okk; n += 1; cell[(h, r['drum'])][0] += okk; cell[(h, r['drum'])][1] += 1
                worst = min(((v[0]/v[1], f'{h}{d}') for (h, d), v in cell.items() if v[1] >= 15))
                line.append(f"{s}: {ok/n*100:.1f}% (最差 {worst[1]} {worst[0]*100:.0f}%)")
            print(f'calib={pk} w={w}', ' | '.join(line), flush=True)


if __name__ == '__main__':
    main()
