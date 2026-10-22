"""Diagnosis on TRUE hits, leave-one-day-out committees. Per take and hand:
  O0 committee only; O1 causal per-drum offset (current method, 2 passes);
  O2 ORACLE global offset (true labels of the whole take; upper bound of any translation correction);
  O3 ORACLE within-take nearest centroid on the 21 rotation dims (leave-one-out; upper bound when today's
     layout is fully known). Oracles use labels and are NOT deployable; they only bound each idea."""
from collections import Counter, defaultdict
import numpy as np
import common as K
from seq_data import build
from fix_sim import _offsets

ROT = slice(83, 104)


def run():
    seq, coms = build(); res = defaultdict(Counter)
    for n, s in seq.items():
        D = s['day']; com, tm = coms[D]['com'], coms[D]['tm']; kind = 'pre' if s['test'] else 'song'
        for h, v in s['hands'].items():
            X, y = v['x'], v['y']
            if len(y) < 20: continue
            p0 = K.proba(com, h, X); o = dict(minh=20, min_drums=1, k_per_drum=1, clamp=0)
            o1, v1 = _offsets(X, p0, tm[h], o); X1 = X.copy(); X1[:, ROT] += o1; p1 = np.where(v1[:, None], K.proba(com, h, X1), p0)
            o2, v2 = _offsets(X, p1, tm[h], o); X2 = X.copy(); X2[:, ROT] += o2; pC = np.where(v2[:, None], K.proba(com, h, X2), p1)
            og = (tm[h][y]-X[:, ROT]).mean(0); Xg = X.copy(); Xg[:, ROT] += og; pG = K.proba(com, h, Xg)
            R = X[:, ROT]; S = np.zeros((len(y), 7)); C = np.zeros(7)
            for d in range(7): S[d] = 0
            sums = np.zeros((7, 21)); cnt = np.bincount(y, minlength=7).astype(float)
            for d in range(7): sums[d] = R[y == d].sum(0)
            nc = np.empty(len(y), int)
            for i in range(len(y)):
                c = sums.copy(); k = cnt.copy(); c[y[i]] -= R[i]; k[y[i]] -= 1; ok = k > 0
                dist = np.full(7, np.inf); dist[ok] = np.linalg.norm(c[ok]/k[ok, None]-R[i], axis=1); nc[i] = dist.argmin()
            for tag, pred in [('O0', p0.argmax(1)), ('O1', pC.argmax(1)), ('O2', pG.argmax(1)), ('O3', nc)]:
                res[(D, kind)][tag] += int((pred == y).sum())
            res[(D, kind)]['n'] += len(y)
    print('day kind      n |  O0 模型  O1 現行校正  O2 理想整體平移  O3 理想今日鼓位')
    for (D, kind), c in sorted(res.items()):
        print(f"{D} {kind:5s} {c['n']:6d} | " + '  '.join(f"{c[t]/c['n']*100:6.1f}" for t in ('O0', 'O1', 'O2', 'O3')))


if __name__ == '__main__':
    run()
