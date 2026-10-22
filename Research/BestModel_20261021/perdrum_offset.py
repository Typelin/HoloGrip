"""Per-drum offset correction. For each candidate drum d, shift the rotation block by drum d's own
offset (template_d - today's centre of d), ask the committee, keep P(d); pick the best d.
  O4 ORACLE: today's centres from true labels (ceiling, not deployable)
  PD causal: centres from EARLIER confident hits of the same take/hand (score >= conf), shrunk toward
     the global offset with k pseudo-hits; drums never seen use the global offset (current method)."""
from collections import Counter, defaultdict
import numpy as np
import common as K
from seq_data import build
from em_layout import corrected
from fix_sim import _offsets

ROT = slice(83, 104)


def score_with_offsets(com, h, X, off):
    """off: (n, 7, 21) per hit and drum -> P(d) using drum d's shifted input."""
    n = len(X); out = np.zeros((n, 7))
    Xs = np.repeat(X[:, None, :], 7, 1).copy(); Xs[:, :, ROT] += off
    P = K.proba(com, h, Xs.reshape(n*7, -1)).reshape(n, 7, 7)
    for d in range(7): out[:, d] = P[:, d, d]
    return out/out.sum(1, keepdims=True)


def causal_offsets(X, p, tm, conf, k, minh=20):
    n = len(X); R = X[:, ROT]; sums = np.zeros((7, 21)); cnt = np.zeros(7); off = np.zeros((n, 7, 21)); valid = np.zeros(n, bool)
    pred = p.argmax(1); ok = p.max(1) >= conf
    for i in range(n):
        if cnt.sum() >= minh:
            m = cnt > 0; g = (cnt[m, None]*tm[m]-sums[m]).sum(0)/cnt[m].sum()
            own = np.where(cnt[:, None] > 0, (tm*cnt[:, None]-sums+k*g)/(cnt[:, None]+k), g)
            off[i] = own; valid[i] = True
        if ok[i]: sums[pred[i]] += R[i]; cnt[pred[i]] += 1
    return off, valid


def main():
    seq, coms = build(); res = defaultdict(Counter); grid = [(c, k) for c in (0.7, 0.85) for k in (2, 5, 10, 30)]
    for n, s in seq.items():
        D = s['day']; kind = 'pre' if s['test'] else 'song'
        for h, v in s['hands'].items():
            X, y = v['x'], v['y']
            if len(y) < 20: continue
            com, tm = coms[D]['com'], coms[D]['tm'][h]; p0, pC = corrected(com, tm, h, X)
            res[(D, kind)]['n'] += len(y); res[(D, kind)]['O1'] += int((pC.argmax(1) == y).sum())
            cen = np.array([X[y == d][:, ROT].mean(0) if (y == d).any() else tm[d] for d in range(7)])
            off = np.repeat((tm-cen)[None], len(y), 0); res[(D, kind)]['O4'] += int((score_with_offsets(com, h, X, off).argmax(1) == y).sum())
            for c, k in grid:
                off, val = causal_offsets(X, pC, tm, c, k); q = np.where(val[:, None], score_with_offsets(com, h, X, off), pC)
                res[(D, kind)][(c, k)] += int((q.argmax(1) == y).sum())
    keys = ['O1', 'O4']+grid
    for k in keys:
        acc = {dk: c[k]/c['n'] for dk, c in res.items()}
        pre = np.mean([v for (d, kd), v in acc.items() if kd == 'pre']); song = np.mean([v for (d, kd), v in acc.items() if kd == 'song'])
        print(str(k).ljust(12), f'前測 {pre*100:5.1f} 歌曲 {song*100:5.1f} |', ' '.join(f"{d}{kd[0]} {acc[(d, kd)]*100:4.1f}" for d, kd in sorted(acc)), flush=True)


if __name__ == '__main__':
    main()
