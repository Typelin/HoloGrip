"""Self-grown drum layout (no labels): inside one take and hand, drums form tight clusters in the 21
rotation dims. Committee probabilities are the prior; per-drum centroids are re-estimated from the
posterior assignments (EM). Batch version = whole take (ceiling, NOT causal); see em_causal for the
time-ordered version. Leave-one-day-out committees, true hits only."""
import sys
from collections import Counter, defaultdict
import numpy as np
import common as K
from seq_data import build
from fix_sim import _offsets

ROT = slice(83, 104)


def corrected(com, tm, h, X):
    p0 = K.proba(com, h, X); o = dict(minh=20, min_drums=1, k_per_drum=1, clamp=0)
    o1, v1 = _offsets(X, p0, tm, o); X1 = X.copy(); X1[:, ROT] += o1; p1 = np.where(v1[:, None], K.proba(com, h, X1), p0)
    o2, v2 = _offsets(X, p1, tm, o); X2 = X.copy(); X2[:, ROT] += o2; return p0, np.where(v2[:, None], K.proba(com, h, X2), p1)


def em(R, prior, beta=1.0, sigma=None, iters=15, min_w=3.0):
    lp = beta*np.log(prior+1e-6); resp = prior.copy()
    for _ in range(iters):
        w = resp.sum(0); mu = (resp.T@R)/np.maximum(w, 1e-9)[:, None]
        d2 = ((R[:, None, :]-mu[None])**2).sum(2)
        s2 = sigma**2 if sigma else float((resp*d2).sum()/resp.sum()/R.shape[1])*R.shape[1]
        ll = lp-d2/(2*s2); ll[:, w < min_w] = lp[:, w < min_w]-50  # no cluster yet -> rely on prior only
        ll[:, w < min_w] = lp[:, w < min_w]-d2[:, w < min_w].mean()/(2*s2)
        ll -= ll.max(1, keepdims=True); resp = np.exp(ll); resp /= resp.sum(1, keepdims=True)
    return resp


def main():
    seq, coms = build(); res = defaultdict(Counter)
    grid = [(b, it) for b in (0.5, 1.0, 2.0) for it in (15,)]
    for n, s in seq.items():
        D = s['day']; kind = 'pre' if s['test'] else 'song'
        for h, v in s['hands'].items():
            X, y = v['x'], v['y']
            if len(y) < 20: continue
            p0, pC = corrected(coms[D]['com'], coms[D]['tm'][h], h, X)
            res[(D, kind)]['O1'] += int((pC.argmax(1) == y).sum()); res[(D, kind)]['n'] += len(y)
            for b, it in grid:
                r = em(X[:, ROT], pC, beta=b, iters=it); res[(D, kind)][f'em{b}'] += int((r.argmax(1) == y).sum())
                r = em(X[:, ROT], p0, beta=b, iters=it); res[(D, kind)][f'em0_{b}'] += int((r.argmax(1) == y).sum())
    keys = ['O1']+[f'em{b}' for b, _ in grid]+[f'em0_{b}' for b, _ in grid]
    print('day kind      n | '+'  '.join(f'{k:>7s}' for k in keys))
    for (D, kind), c in sorted(res.items()):
        print(f"{D} {kind:5s} {c['n']:6d} | "+'  '.join(f"{c[k]/c['n']*100:7.1f}" for k in keys))


if __name__ == '__main__':
    main()
