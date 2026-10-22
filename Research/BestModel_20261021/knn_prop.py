"""Within-take neighbour propagation in the 21 rotation dims, causal: hit i looks at the k most similar
EARLIER hits of the same take/hand (within radius r) and mixes their final drum distributions with its own
committee output: q_i = normalise(p_i^(1-a) * (mean_j q_j + eps)^a). Batch label spreading as ceiling."""
from collections import Counter, defaultdict
import numpy as np
import common as K
from seq_data import build
from em_layout import corrected

ROT = slice(83, 104)


def causal(R, p, k, r, a, eps=0.02):
    n = len(R); q = np.zeros_like(p)
    for i in range(n):
        if i >= 3:
            d = np.linalg.norm(R[:i]-R[i], axis=1); idx = np.argsort(d)[:k]; idx = idx[d[idx] <= r]
            if len(idx) >= 2:
                nb = q[idx].mean(0); v = np.exp((1-a)*np.log(p[i]+1e-6)+a*np.log(nb+eps)); q[i] = v/v.sum(); continue
        q[i] = p[i]
    return q


def spread(R, p, k, a, iters=20):
    d = np.linalg.norm(R[:, None]-R[None], axis=2); np.fill_diagonal(d, np.inf); idx = np.argsort(d, 1)[:, :k]; q = p.copy()
    for _ in range(iters): q = (1-a)*p+a*q[idx].mean(1)
    return q


def main():
    seq, coms = build(); res = defaultdict(Counter)
    grid = [('c', k, r, a) for k in (5, 15) for r in (0.08, 0.15) for a in (0.5, 0.8)]+[('s', k, 0, a) for k in (5, 15) for a in (0.5, 0.9)]
    for n, s in seq.items():
        D = s['day']; kind = 'pre' if s['test'] else 'song'
        for h, v in s['hands'].items():
            X, y = v['x'], v['y']
            if len(y) < 20: continue
            p0, pC = corrected(coms[D]['com'], coms[D]['tm'][h], h, X); R = X[:, ROT]
            res[(D, kind)]['n'] += len(y); res[(D, kind)]['O1'] += int((pC.argmax(1) == y).sum())
            for g in grid:
                q = causal(R, pC, g[1], g[2], g[3]) if g[0] == 'c' else spread(R, pC, g[1], g[3])
                res[(D, kind)][g] += int((q.argmax(1) == y).sum())
    for k in ['O1']+grid:
        acc = {dk: c[k]/c['n'] for dk, c in res.items()}
        pre = np.mean([v for (d, kd), v in acc.items() if kd == 'pre']); song = np.mean([v for (d, kd), v in acc.items() if kd == 'song'])
        print(str(k).ljust(22), f'前測 {pre*100:5.1f} 歌曲 {song*100:5.1f} |', ' '.join(f"{d}{kd[0]} {acc[(d, kd)]*100:4.1f}" for d, kd in sorted(acc)), flush=True)


if __name__ == '__main__':
    main()
