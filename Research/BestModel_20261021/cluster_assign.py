"""Batch ceiling: cluster one take/hand in the 21 rotation dims (GMM), then label clusters with a
one-drum-one-cluster assignment that maximises the committee log-likelihood (Hungarian). Extra
clusters beyond 7 take their own best drum. k = oracle (true #drums) or chosen by BIC. NOT causal."""
from collections import Counter, defaultdict
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.mixture import GaussianMixture
import common as K
from seq_data import build
from em_layout import corrected

ROT = slice(83, 104)


def assign(lab, P, k):
    L = np.log(P+1e-4); C = np.array([L[lab == j].sum(0) for j in range(k)])
    r, c = linear_sum_assignment(-C); m = {a: b for a, b in zip(r, c)}
    for j in range(k):
        if j not in m: m[j] = int(C[j].argmax())
    return np.array([m[j] for j in lab])


def gmm(R, k, seed=0):
    g = GaussianMixture(k, covariance_type='diag', reg_covar=1e-3, n_init=3, random_state=seed).fit(R); return g.predict(R), g.bic(R)


def main():
    seq, coms = build(); res = defaultdict(Counter)
    for n, s in seq.items():
        D = s['day']; kind = 'pre' if s['test'] else 'song'
        for h, v in s['hands'].items():
            X, y = v['x'], v['y']
            if len(y) < 20: continue
            p0, pC = corrected(coms[D]['com'], coms[D]['tm'][h], h, X); R = X[:, ROT]
            kt = len(set(y)); res[(D, kind)]['n'] += len(y); res[(D, kind)]['O1'] += int((pC.argmax(1) == y).sum())
            lab, _ = gmm(R, kt); res[(D, kind)]['k真'] += int((assign(lab, pC, kt) == y).sum())
            best = min((gmm(R, k) + (k,) for k in range(2, 10) if k < len(y)//3), key=lambda t: t[1])
            res[(D, kind)]['kBIC'] += int((assign(best[0], pC, best[2]) == y).sum())
            for k in (8, 10, 12):
                if k < len(y)//3: lab, _ = gmm(R, k); res[(D, kind)][f'k{k}'] += int((assign(lab, pC, k) == y).sum())
                else: res[(D, kind)][f'k{k}'] += int((pC.argmax(1) == y).sum())
    keys = ['O1', 'k真', 'kBIC', 'k8', 'k10', 'k12']
    print('day kind      n | '+'  '.join(f'{k:>6s}' for k in keys))
    for (D, kind), c in sorted(res.items()):
        print(f"{D} {kind:5s} {c['n']:6d} | "+'  '.join(f"{c[k]/c['n']*100:6.1f}" for k in keys))


if __name__ == '__main__':
    main()
