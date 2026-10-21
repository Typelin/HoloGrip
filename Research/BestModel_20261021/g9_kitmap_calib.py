"""Simulation of a short per-day 'kit map' calibration (no training): before playing, each hand taps every
drum K times (instructed, so the drum is known). Those taps give today's position of each drum.
Prediction = model probability combined with distance to today's per-drum calibration centres in the
zero-relative rotation space. Simulated on pretests: the first K hits of every (hand, drum) block act as
the calibration taps and are EXCLUDED from scoring; all other hits are scored as usual."""
import numpy as np, joblib
from collections import defaultdict
from scipy.optimize import linear_sum_assignment
import common as K, cells as Q

ROT = slice(83, 104)


def main():
    D2 = joblib.load(K.OUT/'dataset2.joblib'); T = Q.truth_hands(); com = K.train(D2['rows'], {}, seeds=(0, 1))
    for Kc in (3, 5):
        for w, sig in [(0, 1), (0.5, .15), (0.7, .15), (0.85, .1), (1.0, .15)]:
            cells = defaultdict(lambda: [0, 0])
            for k, truth in T.items():
                items = D2['tests'][k]['items']; tt = np.array([g['t'] for g in truth]); seen = defaultdict(int); calib_ids = set(); cal = defaultdict(list)
                pt = np.array([c['t'] for c in items]); dt = np.abs(tt[:, None]-pt[None, :])
                hm = np.array([[g['hand'] is not None and g['hand'] != c['hand'] for c in items] for g in truth])
                cost = np.hstack([np.where(dt <= 80, dt+300*hm, 1e9), np.full((len(tt), len(tt)), 1e6)])
                r, c = linear_sum_assignment(cost); pairs = {i: j for i, j in zip(r, c) if j < len(pt) and cost[i, j] <= 380}
                order = sorted(range(len(truth)), key=lambda i: truth[i]['t'])
                for i in order:
                    g = truth[i]
                    if g['hand'] is None: continue
                    key = (g['hand'], g['drum'])
                    if seen[key] < Kc and i in pairs:
                        seen[key] += 1; calib_ids.add(i); cal[key].append(items[pairs[i]]['x'][ROT])
                cen = {key: np.mean(v, 0) for key, v in cal.items()}
                for h in 'LR':
                    idx = [j for j, c in enumerate(items) if c['hand'] == h]
                    p = K.proba(com, h, np.stack([items[j]['x'] for j in idx]))
                    if w > 0:
                        dist = np.full((len(idx), 7), 9.)
                        for d_i, d in enumerate(K.DRUMS):
                            if (h, d) in cen: dist[:, d_i] = np.linalg.norm(np.stack([items[j]['x'][ROT] for j in idx])-cen[(h, d)], axis=1)
                        q = np.exp(-dist**2/sig**2); q /= q.sum(1, keepdims=True)+1e-12
                        p = p**(1-w)*(q+1e-6)**w
                    pred = {j: K.DRUMS[int(pp.argmax())] for j, pp in zip(idx, p)}
                    for i, g in enumerate(truth):
                        if g['hand'] != h or i in calib_ids: continue
                        cells[(h, g['drum'])][1] += 1
                        j = pairs.get(i)
                        if j is not None and items[j]['hand'] == h and pred[j] == g['drum']: cells[(h, g['drum'])][0] += 1
            acc = {c: v[0]/max(v[1], 1) for c, v in cells.items()}
            print(f'K={Kc} w={w} sig={sig}', ' '.join(f"{h}{d[:2]}{a*100:.0f}" for (h, d), a in sorted(acc.items())), '| min', round(min(acc.values())*100, 1), 'mean', round(np.mean(list(acc.values()))*100, 1), flush=True)


if __name__ == '__main__':
    main()
