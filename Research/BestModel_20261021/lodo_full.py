"""Leave-one-DAY-out test of the whole 8th-package pipeline (the honest cross-day estimate).
For each day D (round 4, 5, 6, 7): drum committee, drum rotation means and hit gate are trained WITHOUT
any recording of day D; then every recording of day D (pretests, songs, single-hand) is replayed
(batch-equivalent to the field engine). Variants (cumulative):
  A  old rule: peak >= 2.1 g, committee only
  B  hit gate + committee
  C  B + per-drum offset correction
  D  C + pose continuity (600 ms, 16 deg, score < 0.9)
  E  C + pose continuity only when the previous drum is in this hit's top-2
Scores per day: pretest total and 14 cells, songs/single-hand drum rate, hit recall/precision."""
import json, sys
from collections import Counter, defaultdict
import numpy as np, joblib, xgboost as xgb
from scipy.spatial.transform import Rotation as Rot
sys.argv = ['x', '1p5']
import hit_clf as H, common as K, full_report as F
from verify_1023 import TRUTH, match
from fix_sim import _offsets

DAYS = ['R4', 'R5', 'R6', 'R7']


def day_of(name):
    if name.startswith('pretest:round'): return 'R'+name[-1]
    if name in ('seventh_01', 'seventh_12', 'seventh_15'): return 'R7'
    g = K.group_of(name); return 'R7' if g.startswith('R7') else g


def yp(X):
    v = X[:, 83:104].reshape(len(X), 7, 3)[:, 4]*np.pi
    return Rot.from_rotvec(v).as_euler('ZYX', degrees=True)[:, :2]


def run_recording(name, com, tm, gate, variant):
    P = []
    for h in 'LR':
        cs = sorted([c for c in H.CAND[name] if c['hand'] == h], key=lambda c: c['t'])
        if not cs: continue
        if variant == 'A': acc = [c for c in cs if c['g'] >= 2.1]
        else:
            ph = gate.predict_proba(np.stack([H.feats(c) for c in cs]))[:, 1]; acc = [c for c, p in zip(cs, ph) if p >= .4]
        if not acc: continue
        X = np.stack([c['x'] for c in acc]).astype(np.float32); p = K.proba(com, h, X); X2 = X
        if variant in ('C', 'D', 'E'):
            o = dict(minh=20, min_drums=1, k_per_drum=1, clamp=0)
            o1, v1 = _offsets(X, p, tm[h], o); X1 = X.copy(); X1[:, 83:104] += o1; p1 = np.where(v1[:, None], K.proba(com, h, X1), p)
            o2, v2 = _offsets(X, p1, tm[h], o); X2 = X.copy(); X2[:, 83:104] += o2; p = np.where(v2[:, None], K.proba(com, h, X2), p1)
        lab = p.argmax(1).copy(); sc = p.max(1); Y = yp(X2); t = np.array([c['t'] for c in acc])
        if variant in ('D', 'E'):
            for i in range(1, len(lab)):
                if sc[i] < .9 and t[i]-t[i-1] <= 600 and np.hypot(*(Y[i]-Y[i-1])) <= 16:
                    if variant == 'D' or lab[i-1] in np.argsort(-p[i])[:2]: lab[i] = lab[i-1]
        P += [dict(t=ti, peak_ms=ti, hand=h, drum=K.DRUMS[int(l)]) for ti, l in zip(t, lab)]
    return P


def main():
    D2 = joblib.load(K.OUT/'dataset2.joblib'); rows = D2['rows']
    names = list(H.CAND); L = {n: H.labels(n) for n in names}
    out = {}
    for D in DAYS:
        tr = [r for r in rows if day_of(r['session']) != D]
        com = K.train(tr, {}, seeds=(0, 1))
        tm = {h: np.array([np.mean([r['x'][83:104] for r in tr if r['hand'] == h and r['drum'] == d], 0) if any(r['hand'] == h and r['drum'] == d for r in tr) else np.zeros(21) for d in K.DRUMS]) for h in 'LR'}
        gn = [n for n in names if n not in K.TESTS and day_of(n) != D]
        Xg = np.vstack([np.stack([H.feats(c) for c in H.CAND[n]])[L[n][0] != 2] for n in gn]); yg = np.concatenate([L[n][0][L[n][0] != 2] for n in gn])
        gate = xgb.XGBClassifier(n_estimators=800, max_depth=6, learning_rate=.05, subsample=.8, colsample_bytree=.6, device='cuda', tree_method='hist', random_state=0).fit(Xg, yg)
        test_names = [n for n in names if day_of(n) == D]
        out[D] = {}
        for V in 'ABCDE':
            cells = defaultdict(Counter); pre = Counter(); song = Counter()
            for n in test_names:
                P = run_recording(n, com, tm, gate, V)
                if n in K.TESTS:
                    truth = TRUTH[n]; pairs = match(truth, P)
                    for i, g in enumerate(truth):
                        j = pairs.get(i); ok = j is not None and (g['hand'] is None or P[j]['hand'] == g['hand']) and P[j]['drum'] == g['drum']
                        pre['ok'] += ok; pre['n'] += 1
                        if g['hand']: cells[(g['hand'], g['drum'])]['ok'] += ok; cells[(g['hand'], g['drum'])]['n'] += 1
                else:
                    u, e = F.truth_of(n); res, _ = F.score(u, e, P)
                    song['ok'] += res.get('correct', 0); song['n'] += res['midi']; song['extra'] += res['extra']; song['matched'] += res['matched']; song['out'] += res['detected']-res.get('near_excluded', 0)
            acc = {f'{h}{d}': round(v['ok']/v['n'], 3) for (h, d), v in cells.items()}
            rec = dict(pretest=round(pre['ok']/max(pre['n'], 1), 4), cells=acc, cell_mean=round(float(np.mean(list(acc.values()))), 4) if acc else None,
                       songs=round(song['ok']/max(song['n'], 1), 4), hit_recall=round(song['matched']/max(song['n'], 1), 4),
                       hit_precision=round(song['matched']/max(song['matched']+song['extra'], 1), 4), extra=song['extra'])
            out[D][V] = rec
            print(D, V, '| 前測', rec['pretest'], '14格', rec['cell_mean'], '| 歌曲單手', rec['songs'], '打擊抓到', rec['hit_recall'], '正確', rec['hit_precision'], '多判', rec['extra'], flush=True)
        (K.OUT/'lodo_full.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
