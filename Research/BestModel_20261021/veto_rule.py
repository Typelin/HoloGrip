"""Exclusivity veto (causal, no labels): each drum's centre in the 21 rotation dims is learned from EARLIER
hits of the same take/hand that were labelled with confidence >= conf. If this hit's top drum already has
a centre (>= m hits) and the hit is farther than r_far from it, while the next candidate (top-3, P >= pmin)
has no centre yet or is within r_near, take that candidate. Ties: nearest established centre."""
from collections import Counter, defaultdict
import numpy as np
import common as K
from seq_data import build
from em_layout import corrected

ROT = slice(83, 104)


def veto(R, p, m, r_far, r_near, pmin, conf=0.7, top=3):
    n = len(R); sums = np.zeros((7, 21)); cnt = np.zeros(7); lab = p.argmax(1).copy(); changed = 0
    for i in range(n):
        order = np.argsort(-p[i])[:top]; d = order[0]
        cen = sums/np.maximum(cnt, 1)[:, None]; est = cnt >= m
        if est[d] and np.linalg.norm(R[i]-cen[d]) > r_far:
            for d2 in order[1:]:
                if p[i, d2] < pmin: break
                if not est[d2] or np.linalg.norm(R[i]-cen[d2]) <= r_near:
                    lab[i] = d2; changed += 1; break
        if p[i].max() >= conf or lab[i] != d: sums[lab[i]] += R[i]; cnt[lab[i]] += 1
    return lab, changed


def main():
    seq, coms = build(); res = defaultdict(Counter); cache = []
    for n, s in seq.items():
        D = s['day']; kind = 'pre' if s['test'] else 'song'
        for h, v in s['hands'].items():
            if len(v['y']) < 20: continue
            p0, pC = corrected(coms[D]['com'], coms[D]['tm'][h], h, v['x']); cache.append((D, kind, v['x'][:, ROT], pC, v['y']))
    grid = [(m, rf, rn, pm) for m in (5, 10) for rf in (0.35, 0.5, 0.7) for rn in (0.3, 0.45) for pm in (0.05, 0.15)]
    for D, kind, R, pC, y in cache:
        res[(D, kind)]['n'] += len(y); res[(D, kind)]['O1'] += int((pC.argmax(1) == y).sum())
        for g in grid: lab, _ = veto(R, pC, *g); res[(D, kind)][g] += int((lab == y).sum())
    out = []
    for k in ['O1']+grid:
        acc = {dk: c[k]/c['n'] for dk, c in res.items()}
        pre = np.mean([v for (d, kd), v in acc.items() if kd == 'pre']); song = np.mean([v for (d, kd), v in acc.items() if kd == 'song'])
        out.append((k, pre, song, acc))
    for k, pre, song, acc in [out[0]]+sorted(out[1:], key=lambda r: -(r[1]+r[2]))[:10]:
        print(str(k).ljust(24), f'前測 {pre*100:5.1f} 歌曲 {song*100:5.1f} |', ' '.join(f"{d}{kd[0]} {acc[(d, kd)]*100:4.1f}" for d, kd in sorted(acc)))


if __name__ == '__main__':
    main()
