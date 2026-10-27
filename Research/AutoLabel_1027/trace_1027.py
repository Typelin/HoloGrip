"""Hit-by-hit trace for one pretest hand (cached full model, best setting): true drum, final label,
model top-2 with scores, whether continuity changed it, corrected snare-relative pose."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import exp_1027 as E
L5, K = E.L5, E.K
BEST = dict(E.DEFAULT, w_rc=0.6, anchor_top=1, conf=0.8)


def trace(tag, nm, h, drums, o=BEST):
    M = E.load(tag); d = M['items'][nm][h]; X, t = d['X'], d['t']; n = len(X); w = o['w_rc']; tm = M['tm'][h]
    p0 = E.proba(M, h, X, w)
    anchor = set()
    if n >= 5 and all(E.SNARE in np.argsort(-p0[i])[:o['anchor_top']] for i in range(5)):
        Y = L5.LF.yp(X[:5]); med = np.median(Y, 0)
        if np.all(np.hypot(*(Y-med).T) <= o['anchor_deg']): anchor = set(range(5))
    p = p0
    for _ in range(2):
        pred = p.argmax(1); ok = p.max(1) >= o['conf']; sums = np.zeros((7, 21)); cnt = np.zeros(7); na = 0; Xc = X.copy(); valid = np.zeros(n, bool)
        for i in range(n):
            if cnt.sum() >= o['minh'] or na >= 5: Xc[i, E.ROT] = E.correction(sums, cnt, tm, o)(X[i, E.ROT]); valid[i] = True
            if i in anchor: sums[E.SNARE] += X[i, E.ROT]; cnt[E.SNARE] += 1; na += 1
            elif ok[i]: sums[pred[i]] += X[i, E.ROT]; cnt[pred[i]] += 1
        p = np.where(valid[:, None], E.proba(M, h, Xc, w), p)
    lab = p.argmax(1).copy(); sc = p.max(1); Y = L5.LF.yp(Xc); chg = np.zeros(n, bool)
    for i in range(1, n):
        if sc[i] < o['cont_score'] and t[i]-t[i-1] <= 600 and np.hypot(*(Y[i]-Y[i-1])) <= o['cont_deg'] and lab[i-1] in np.argsort(-p[i])[:o['cont_top']]:
            chg[i] = lab[i] != lab[i-1]; lab[i] = lab[i-1]
    tmY = {K.DRUMS[i]: L5.LF.yp(np.r_[np.zeros(83), tm[i]][None].astype(np.float32))[0] for i in range(7)}
    print(f'== {tag} {nm} {h}  anchor={bool(anchor)}  訓練鼓位(yaw,pitch):', {k[:2]: v.round(0).tolist() for k, v in tmY.items()})
    T = [g for g in M['truth'][nm] if g['hand'] == h]
    for g in T:
        if g['drum'] not in drums: continue
        j = int(np.argmin(np.abs(t-g['t'])))
        if abs(t[j]-g['t']) > 80: print('  miss', g['drum']); continue
        top = np.argsort(-p[j])[:2]
        print(f"  真={g['drum'][:2]} 判={K.DRUMS[lab[j]][:2]}{'(連續)' if chg[j] else ''}  前二={K.DRUMS[top[0]][:2]}{p[j][top[0]]:.2f}/{K.DRUMS[top[1]][:2]}{p[j][top[1]]:.2f}  校正後={Y[j].round(0).tolist()}")


if __name__ == '__main__':
    trace(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4].split(','))
