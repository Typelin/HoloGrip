"""Leave-one-day-out song / single-hand score WITH the field snare anchor (plain vs re-centred training).

The field Take always seeds the per-drum offset with each hand's first 5 accepted hits as snare
(players tap the snare first). Songs do not start with taps, so here the anchor is emulated by the
hand's first 5 accepted hits that are MIDI snare hits (equivalent to tapping the snare 5 times
before the song); every other hit is handled exactly like the field (causal offset + top-2 continuity).
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import final_1027_rc as RC
FB, L5, K, H, LA, F = RC.FB, RC.L5, RC.K, RC.H, RC.FB.LA, RC.FB.F
SNARE = K.DRUMS.index('小鼓')


def offsets(X, p, tm, anchor, conf=.7, minh=20, min_anchor=5):
    n = len(X); pred = p.argmax(1); ok = p.max(1) >= conf
    sums = np.zeros((7, 21)); cnt = np.zeros(7); na = 0; out = np.zeros((n, 21)); valid = np.zeros(n, bool)
    for i in range(n):
        m = cnt > 0
        if cnt[m].sum() >= minh or na >= min_anchor:
            out[i] = (cnt[m, None]*tm[m]-sums[m]).sum(0)/cnt[m].sum(); valid[i] = True
        if i in anchor: sums[SNARE] += X[i, 83:104]; cnt[SNARE] += 1; na += 1
        elif ok[i]: sums[pred[i]] += X[i, 83:104]; cnt[pred[i]] += 1
    return out, valid


def run(name, com, tm, gate, used):
    P = []
    for h in 'LR':
        cs = sorted([c for c in H.CAND[name] if c['hand'] == h], key=lambda c: c['t'])
        if not cs: continue
        ph = gate.predict_proba(np.stack([H.feats(c) for c in cs]))[:, 1]; acc = [c for c, q in zip(cs, ph) if q >= .4]
        if not acc: continue
        sn = np.array(sorted(u['t'] for u in used if u['hand'] == h and u['drum'] == '小鼓'))
        anchor = set()
        for i, c in enumerate(acc):
            if len(sn) and np.abs(sn-c['t']).min() <= 60: anchor.add(i)
            if len(anchor) == 5: break
        X = np.stack([c['x'] for c in acc]).astype(np.float32); p0 = K.proba(com, h, X)
        o1, v1 = offsets(X, p0, tm[h], anchor); X1 = X.copy(); X1[:, 83:104] += o1; p1 = np.where(v1[:, None], K.proba(com, h, X1), p0)
        o2, v2 = offsets(X, p1, tm[h], anchor); X2 = X.copy(); X2[:, 83:104] += o2; p = np.where(v2[:, None], K.proba(com, h, X2), p1)
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
            ok = n = 0; cells = {}
            for nm in [x for x in names if L5.day_of(x) == D and x not in K.TESTS]:
                used, excl = T8[nm] if nm in T8 else F.truth_of(nm)
                r, cl = F.score(used, excl, run(nm, com, tm, gate, used))
                ok += r.get('correct', 0); n += r['midi']
                for g, o in cl:
                    if g['hand']:
                        c = cells.setdefault(f"{g['hand']}|{g['drum']}", [0, 0]); c[0] += o == 'ok'; c[1] += 1
            res[f'{D}|{variant}'] = dict(ok=ok, n=n, rate=round(ok/max(n, 1), 4), cells=cells)
            print(D, variant, '歌曲＋單手（有小鼓錨點）', ok, '/', n, round(ok/max(n, 1), 4), flush=True)
            (FB.OUT/'song_anchor_1027.json').write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
