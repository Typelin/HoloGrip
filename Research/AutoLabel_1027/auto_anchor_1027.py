"""Leave-one-day-out with an AUTOMATIC snare anchor (no cooperation assumed), plain vs re-centred training.

Per hand, the first 5 accepted hits are used as the snare anchor only if
  * for each of them the uncorrected committee ranks 小鼓 in its top-2, and
  * their zero-relative poses (yaw/pitch at the hit) all lie within 8 deg of their median.
Otherwise there is no anchor and the per-drum offset starts after 20 confident hits (field fallback).
Scored on pretests (most start with snare taps; round 4 starts with Hi-Hat) and on songs/single-hand
(no taps). Everything else is the field logic (causal offset, 2 passes, top-2 continuity).
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import final_1027_rc as RC
import song_anchor_1027 as SA
FB, L5, K, H, F = RC.FB, RC.L5, RC.K, RC.H, RC.FB.F
SNARE = K.DRUMS.index('小鼓')


def auto_anchor(X, p0, n=5, top=2, max_deg=8.0):
    if len(X) < n: return set()
    if not all(SNARE in np.argsort(-p0[i])[:top] for i in range(n)): return set()
    Y = L5.LF.yp(X[:n]); med = np.median(Y, 0)
    return set(range(n)) if np.all(np.hypot(*(Y-med).T) <= max_deg) else set()


def run(name, com, tm, gate, stat):
    P = []
    for h in 'LR':
        cs = sorted([c for c in H.CAND[name] if c['hand'] == h], key=lambda c: c['t'])
        if not cs: continue
        ph = gate.predict_proba(np.stack([H.feats(c) for c in cs]))[:, 1]; acc = [c for c, q in zip(cs, ph) if q >= .4]
        if not acc: continue
        X = np.stack([c['x'] for c in acc]).astype(np.float32); p0 = K.proba(com, h, X)
        anchor = auto_anchor(X, p0); stat['anchored' if anchor else 'not'] += 1
        o1, v1 = SA.offsets(X, p0, tm[h], anchor); X1 = X.copy(); X1[:, 83:104] += o1; p1 = np.where(v1[:, None], K.proba(com, h, X1), p0)
        o2, v2 = SA.offsets(X, p1, tm[h], anchor); X2 = X.copy(); X2[:, 83:104] += o2; p = np.where(v2[:, None], K.proba(com, h, X2), p1)
        lab = p.argmax(1).copy(); sc = p.max(1); Y = L5.LF.yp(X2); t = np.array([c['t'] for c in acc])
        for i in range(1, len(lab)):
            if sc[i] < .9 and t[i]-t[i-1] <= 600 and np.hypot(*(Y[i]-Y[i-1])) <= 16 and lab[i-1] in np.argsort(-p[i])[:2]: lab[i] = lab[i-1]
        P += [dict(t=ti, peak_ms=ti, hand=h, drum=K.DRUMS[int(l)]) for ti, l in zip(t, lab)]
    return P


def main():
    rows, names, Lb, T8, R8 = FB.setup()
    res = {}
    for D in FB.DAYS:
        for variant in ('plain', 'rc'):
            if variant == 'plain': com, tm, gate, _ = L5.train_day(rows, names, Lb, {D})
            else: com, tm, gate, _, _ = RC.train_day_rc(rows, names, Lb, {D})
            pre = dict(ok=0, n=0, cells={}); song = dict(ok=0, n=0, cells={}); st = {'pre': dict(anchored=0, **{'not': 0}), 'song': dict(anchored=0, **{'not': 0})}
            for nm in [x for x in names if L5.day_of(x) == D]:
                if nm in K.TESTS:
                    P = run(nm, com, tm, gate, st['pre']); truth = L5.TRUTH[nm]; pairs = L5.match(truth, P)
                    for i, g in enumerate(truth):
                        j = pairs.get(i); ok = j is not None and (g['hand'] is None or P[j]['hand'] == g['hand']) and P[j]['drum'] == g['drum']
                        pre['ok'] += ok; pre['n'] += 1
                        if g['hand']: c = pre['cells'].setdefault(f"{g['hand']}|{g['drum']}", [0, 0]); c[0] += ok; c[1] += 1
                else:
                    used, excl = T8[nm] if nm in T8 else F.truth_of(nm)
                    r, cl = F.score(used, excl, run(nm, com, tm, gate, st['song']))
                    song['ok'] += r.get('correct', 0); song['n'] += r['midi']
                    for g, o in cl:
                        if g['hand']: c = song['cells'].setdefault(f"{g['hand']}|{g['drum']}", [0, 0]); c[0] += o == 'ok'; c[1] += 1
            res[f'{D}|{variant}'] = dict(pretest=pre, songs=song, anchor_use=st)
            print(D, variant, '自動錨點 | 前測', pre['ok'], '/', pre['n'], round(pre['ok']/max(pre['n'], 1), 4), '| 歌曲單手', song['ok'], '/', song['n'],
                  round(song['ok']/max(song['n'], 1), 4), '| 錨點啟用', st, flush=True)
            (FB.OUT/'auto_anchor_1027.json').write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
