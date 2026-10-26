"""Fast, batch-equivalent simulation of the current 8th package (1.5 g candidates -> hit gate ->
committee -> per-drum offset, 2 passes), plus post rules that use only earlier hits of the same hand:
  R1 pose continuity: if this hit's score < s and the previous accepted hit of the same hand was within
     T ms and its zero-relative yaw/pitch is within D deg, take the previous hit's drum.
  R2 local vote: among the last k accepted hits of the same hand within W ms whose pose is within D deg
     (plus this one), take the score-weighted majority drum.
  R3 abstain: hits with final score < a are not given a drum (reported separately).
Scores: pretest 14 cells (+ answered share for R3) and songs/single takes (field scoring)."""
import json, sys, itertools
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np, joblib
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'第八次的包_暫定2026-10-27/Program'))
import seventh_core as core
import venue_free_runtime_1023 as R
from scipy.spatial.transform import Rotation as Rot
from verify_1023 import TRUTH, match
import full_report as F, common as K
from fix_sim import _offsets, SONGS, SINGLES

CAND = joblib.load(K.OUT/'hit_cand_1p5.joblib')


def yp(X):
    v = X[:, 83:104].reshape(len(X), 7, 3)[:, 4]*np.pi
    return Rot.from_rotvec(v).as_euler('ZYX', degrees=True)[:, :2]


def base_run(bundle, name):
    """Per hand, time-ordered accepted hits with final probabilities (current package)."""
    tm = {h: np.array([bundle['drum_rotation_means'][f'{h}|{d}'] for d in bundle['classes']]) for h in 'LR'}
    o = dict(minh=20, min_drums=1, k_per_drum=1, clamp=0); out = {}
    for h in 'LR':
        cs = sorted([c for c in CAND[name] if c['hand'] == h], key=lambda c: c['t'])
        if not cs: continue
        G = np.stack([np.r_[c['g'], c['det'] if len(c['det']) else np.zeros(18), c['x'], c['ctx']] for c in cs]).astype(np.float32)
        ph = bundle['_hit_gate'].predict(bundle['_xgb'].DMatrix(G))
        acc = [c for c, p in zip(cs, ph) if p >= .4]
        if not acc: continue
        X = np.stack([c['x'] for c in acc]).astype(np.float32); p0 = R.probabilities(bundle, X, h)
        o1, v1 = _offsets(X, p0, tm[h], o); X1 = X.copy(); X1[:, 83:104] += o1; p1 = np.where(v1[:, None], R.probabilities(bundle, X1, h), p0)
        o2, v2 = _offsets(X, p1, tm[h], o); X2 = X.copy(); X2[:, 83:104] += o2; pf = np.where(v2[:, None], R.probabilities(bundle, X2, h), p1)
        out[h] = dict(t=np.array([c['t'] for c in acc]), p=pf, yp=yp(X2), classes=list(bundle['classes']))
    return out


def apply_rules(run, r):
    P = []
    for h, d in run.items():
        lab = d['p'].argmax(1).copy(); sc = d['p'].max(1)
        for i in range(len(lab)):
            if r.get('R1') and i > 0:
                T, D, s = r['R1']
                j = i-1
                if sc[i] < s and d['t'][i]-d['t'][j] <= T and np.hypot(*(d['yp'][i]-d['yp'][j])) <= D: lab[i] = lab[j]
            if r.get('R2'):
                k, W, D = r['R2']; js = [j for j in range(max(0, i-k), i) if d['t'][i]-d['t'][j] <= W and np.hypot(*(d['yp'][i]-d['yp'][j])) <= D]
                if js:
                    v = d['p'][i].copy()
                    for j in js: v[lab[j]] += sc[j]
                    lab[i] = int(v.argmax())
        for i in range(len(lab)):
            if r.get('R3') and sc[i] < r['R3']: P.append(dict(t=d['t'][i], peak_ms=d['t'][i], hand=h, drum=None)); continue
            P.append(dict(t=d['t'][i], peak_ms=d['t'][i], hand=h, drum=d['classes'][int(lab[i])]))
    return P


def evaluate(runs, r):
    cells = defaultdict(Counter); ans = tot = 0
    for k in K.TESTS:
        P = apply_rules(runs[k], r); truth = TRUTH[k]; pairs = match(truth, P)
        for i, g in enumerate(truth):
            if not g['hand']: continue
            j = pairs.get(i); o = 'x'
            if j is not None and P[j]['hand'] == g['hand']:
                if P[j]['drum'] is None: o = 'abstain'
                else: o = 'ok' if P[j]['drum'] == g['drum'] else 'x'; ans += 1
            cells[(g['hand'], g['drum'])][o] += 1; tot += 1
    acc = {c: v['ok']/sum(v.values()) for c, v in cells.items()}
    prec = sum(v['ok'] for v in cells.values())/max(ans, 1)
    song = Counter()
    for k in SONGS+SINGLES:
        u, e = F.truth_of(k); P = [p for p in apply_rules(runs[k], r) if p['drum'] is not None]; res, _ = F.score(u, e, P)
        song['c'] += res.get('correct', 0); song['m'] += res['midi']; song['w'] += res.get('wrong_drum', 0)
    return acc, prec, ans/tot, song


def main():
    bundle, _ = core.load_components()
    runs = {k: base_run(bundle, k) for k in K.TESTS+SONGS+SINGLES}
    rules = [('目前第八次包', {})]
    rules += [(f'R1 姿勢連續 {T}ms/{D}°/分數<{s}', dict(R1=(T, D, s))) for T, D, s in itertools.product([300, 600], [8, 12], [0.7, 0.9])]
    rules += [(f'R2 鄰近投票 k{k}/{W}ms/{D}°', dict(R2=(k, W, D))) for k, W, D in itertools.product([3, 6], [600, 1500], [6, 10])]
    rules += [(f'R3 沒把握不回答 <{a}', dict(R3=a)) for a in (0.5, 0.6, 0.7, 0.8)]
    out = []
    for name, r in rules:
        acc, prec, answered, song = evaluate(runs, r)
        rec = dict(name=name, mean=round(float(np.mean(list(acc.values()))), 4), min=round(min(acc.values()), 4), precision=round(prec, 4), answered=round(answered, 4),
                   song=round(song['c']/song['m'], 4), song_wrong=song['w'], cells={f'{h}{d}': round(v, 3) for (h, d), v in acc.items()})
        out.append(rec)
        print(f"{name:32s} | 14格 {rec['mean']:.4f} 最低 {rec['min']:.4f} | 回答的準確 {rec['precision']:.4f} 回答率 {rec['answered']:.3f} | 歌曲單手 {rec['song']:.4f} 錯鼓 {rec['song_wrong']} | RCr {acc[('R','Crash')]:.2f} RRi {acc[('R','Ride')]:.2f} LRi {acc[('L','Ride')]:.2f}", flush=True)
    (K.OUT/'rules_sim.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
