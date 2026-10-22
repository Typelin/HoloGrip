"""Structured layout EM (batch ceiling first): today's drum centre mu_d = template_d + s + e_d, with a
take-wide shift s (free) and per-drum deviation e_d ~ N(0, tau^2) (shrunk to 0 with few hits).
Likelihood per hit: beta*log committee + Gaussian(rotation | mu_d, sigma^2). No labels used."""
from collections import Counter, defaultdict
import numpy as np
import common as K
from seq_data import build
from em_layout import corrected

ROT = slice(83, 104)


def pooled_sigma(seq):
    v = []
    for s in seq.values():
        for h, d in s['hands'].items():
            for c in range(7):
                m = d['y'] == c
                if m.sum() >= 10: v.append(d['x'][m][:, ROT].var(0))
    return float(np.sqrt(np.mean(v)))


def sem(R, prior, tm, beta, tau, sigma, iters=20, shift0=None):
    lp = beta*np.log(prior+1e-6); s = np.zeros(21) if shift0 is None else shift0; e = np.zeros((7, 21)); resp = prior.copy()
    for it in range(iters):
        mu = tm+s+e; d2 = ((R[:, None, :]-mu[None])**2).sum(2); ll = lp-d2/(2*sigma**2)
        ll -= ll.max(1, keepdims=True); resp = np.exp(ll); resp /= resp.sum(1, keepdims=True)
        w = resp.sum(0); sumR = resp.T@R
        # shift: weighted mean residual after per-drum deviation
        s = ((sumR-w[:, None]*(tm+e)).sum(0))/max(w.sum(), 1e-9)
        # per-drum deviation, MAP with Gaussian prior tau
        k = (sigma/tau)**2 if tau else np.inf
        e = (sumR-w[:, None]*(tm+s))/(w[:, None]+k) if tau else np.zeros((7, 21))
    return resp


def main():
    seq, coms = build(); sig = pooled_sigma(seq); print('pooled within-drum sigma per dim', round(sig, 4))
    grid = [(b, t, f) for b in (0.3, 1.0) for t in (0.0, 0.05, 0.1, 0.2) for f in (1.0, 2.0)]
    res = defaultdict(Counter)
    for n, s in seq.items():
        D = s['day']; kind = 'pre' if s['test'] else 'song'
        for h, v in s['hands'].items():
            X, y = v['x'], v['y']
            if len(y) < 20: continue
            tm = coms[D]['tm'][h]; p0, pC = corrected(coms[D]['com'], tm, h, X); R = X[:, ROT]
            res[(D, kind)]['n'] += len(y); res[(D, kind)]['O1'] += int((pC.argmax(1) == y).sum())
            for b, t, f in grid:
                r = sem(R, p0, tm, b, t, sig*f); res[(D, kind)][(b, t, f)] += int((r.argmax(1) == y).sum())
    keys = ['O1']+grid
    rows = []
    for k in keys:
        acc = {dk: c[k]/c['n'] for dk, c in res.items()}
        pre = np.mean([v for (d, kd), v in acc.items() if kd == 'pre']); song = np.mean([v for (d, kd), v in acc.items() if kd == 'song'])
        rows.append((k, pre, song, acc))
    for k, pre, song, acc in sorted(rows, key=lambda r: -(r[1]+r[2])):
        print(str(k).ljust(18), f'前測平均 {pre*100:5.1f} 歌曲平均 {song*100:5.1f} |', ' '.join(f"{d}{kd[0]} {acc[(d, kd)]*100:4.1f}" for d, kd in sorted(acc)))


if __name__ == '__main__':
    main()
