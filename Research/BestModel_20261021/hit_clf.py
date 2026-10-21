"""Hit-detection study, step 2: label candidates from MIDI and learn hit / not-hit.
Labels (per hand): candidate matched one-to-one (<= 60 ms) to a used MIDI of the same hand -> hit;
unmatched candidate within 80 ms of an excluded / unlabelled / hand-unknown MIDI -> ambiguous (ignored);
other unmatched candidates -> not a hit (other hand's hit, rebound, arm motion).
Scores per recording: recall = used MIDI found / used MIDI; precision = kept hits / kept (non-ambiguous).
Baseline = peak >= 2.1 g. Classifier = XGBoost (GPU) on detector shape + 104 features + context,
evaluated leave-one-day-out on songs/single takes and trained on everything for the pretests."""
import json, sys
from collections import defaultdict
import numpy as np, joblib
import xgboost as xgb
from scipy.optimize import linear_sum_assignment
import common as K, full_report as F
from verify_1023 import TRUTH

TAG = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[0].endswith('-c') else '1p5'
CAND = joblib.load(K.OUT/f'hit_cand_{TAG}.joblib')
DAY = {'fourth_song03': 'R4', 'pretest:round4': 'T', 'pretest:round5': 'T', 'pretest:round6': 'T', 'seventh_01': 'T', 'seventh_12': 'T', 'seventh_15': 'T'}


def day(n):
    if n in DAY: return DAY[n]
    return K.group_of(n)


def labels(name):
    cands = CAND[name]
    if name in K.TESTS: used = [g for g in TRUTH[name] if g['hand']]; amb_t = [g['t'] for g in TRUTH[name] if not g['hand']]
    else:
        used, amb_t = F.truth_of(name)
    lab = np.full(len(cands), -1); ut = np.array([u['t'] for u in used]); found = np.zeros(len(used), bool)
    allt = np.r_[ut, np.array(amb_t, float)]; lo, hi = allt.min()-200, allt.max()+200
    for h in 'LR':
        ci = [i for i, c in enumerate(cands) if c['hand'] == h]; ui = [j for j, u in enumerate(used) if u['hand'] == h]
        if ci and ui:
            dt = np.abs(np.array([cands[i]['t'] for i in ci])[:, None]-ut[ui][None, :]); cost = np.where(dt <= 60, dt, 1e9)
            r, c = linear_sum_assignment(cost)
            for a, b in zip(r, c):
                if cost[a, b] <= 60: lab[ci[a]] = 1; found[ui[b]] = True
    amb = np.array(amb_t) if len(amb_t) else np.array([1e18])
    for i, c in enumerate(cands):
        if lab[i] == 1: continue
        lab[i] = 2 if (np.min(np.abs(amb-c['t'])) <= 80 or not lo <= c['t'] <= hi) else 0
    return lab, len(used)


def feats(c):
    d = c['det'] if len(c['det']) else np.zeros(18, np.float32)
    return np.r_[c['g'], d, c['x'], c['ctx']].astype(np.float32)


def score(name, keep, lab, n_used):
    kp = keep & (lab == 1); kn = keep & (lab == 0)
    rec = kp.sum()/max(n_used, 1); prec = kp.sum()/max(kp.sum()+kn.sum(), 1)
    return dict(recall=round(float(rec), 4), precision=round(float(prec), 4), used=n_used, found=int(kp.sum()), extra=int(kn.sum()), cand_hits=int((lab == 1).sum()))


def main():
    names = list(CAND); L = {n: labels(n) for n in names}; X = {n: np.stack([feats(c) for c in CAND[n]]) for n in names}
    G = {n: np.array([c['g'] for c in CAND[n]]) for n in names}
    print('candidate coverage (MIDI with any 1.5 g candidate):', {n: round(float((L[n][0] == 1).sum()/max(L[n][1], 1)), 3) for n in names}, flush=True)
    train_names = [n for n in names if n not in K.TESTS]
    res = defaultdict(dict)
    for n in names:   # baseline
        res['base_2.1g'][n] = score(n, G[n] >= 2.1, *L[n])
    def fit(ns, seed=0):
        Xt = np.vstack([X[n][L[n][0] != 2] for n in ns]); yt = np.concatenate([L[n][0][L[n][0] != 2] for n in ns])
        m = xgb.XGBClassifier(n_estimators=800, max_depth=6, learning_rate=.05, subsample=.8, colsample_bytree=.6, device='cuda', tree_method='hist', random_state=seed)
        m.fit(Xt, yt); return m
    prob = {}
    for d in ['R4', 'R5', 'R6', 'R7o', 'R7m']:
        m = fit([n for n in train_names if day(n) != d])
        for n in train_names:
            if day(n) == d: prob[n] = m.predict_proba(X[n])[:, 1]
    m = fit(train_names)
    for n in K.TESTS: prob[n] = m.predict_proba(X[n])[:, 1]
    for th in (0.3, 0.4, 0.5, 0.6, 0.7):
        for n in names: res[f'clf_{th}'][n] = score(n, prob[n] >= th, *L[n])
    for key, r in res.items():
        def agg(ns):
            f = sum(r[n]['found'] for n in ns); u = sum(r[n]['used'] for n in ns); e = sum(r[n]['extra'] for n in ns)
            return f'recall {f/u:.4f} precision {f/max(f+e,1):.4f} (extra {e})'
        print(key, '| 訓練錄音(換天考)', agg(train_names), '| 前測', agg(K.TESTS), flush=True)
    joblib.dump(dict(res=res, prob=prob), K.OUT/f'hit_clf_{TAG}.joblib')
    (K.OUT/f'hit_clf_{TAG}.json').write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
