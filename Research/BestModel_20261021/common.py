"""Shared data + training + scoring for the best-model search (rounds 4-7).

Exams
  * Pretests (never trained): round4/5/6, today 01 (original kit), 12 & 15 (moved kit).
    Full pipeline on detector peaks (2.1 g), one-to-one time match within 80 ms, drum must match.
  * Group-out cross validation on training data: R4, R5, R6, R7 original kit, R7 moved kit.
    Train without that group, classify its labelled hits (hand known), report accuracy.
Strict standard: features are causal per hit (-240..+80 ms + recorded drumhead zero), no compass,
pretests never in training, all compute on GPU.
"""
from __future__ import annotations
import json, sys, time
from collections import Counter, defaultdict
from pathlib import Path

import joblib
import numpy as np
import torch
import xgboost as xgb
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[2]
P6 = ROOT/'第六次的包_暫定2026-10-16'; P7 = ROOT/'第七次的包_暫定2026-10-20'
sys.path[:0] = [str(P6/'Program'), str(P7/'Program'), str(ROOT/'Research/AutoLabel_v2_20261021')]
OUT = Path(__file__).resolve().parent/'out'; OUT.mkdir(exist_ok=True)
EXP = P7/'Data/實驗_姿態模型自動標記_20261020'
LAB = P7/'Data/自動手別v2_預填匯入_20261021'
DRUMS = ['小鼓', '高音 Tom', '中音 Tom', '落地 Tom', 'Hi-Hat', 'Crash', 'Ride']
TESTS = ['pretest:round4', 'pretest:round5', 'pretest:round6', 'seventh_01', 'seventh_12', 'seventh_15']
DEV = 'cuda'


def group_of(session):
    if session.startswith('fourth'): return 'R4'
    if session.startswith('fifth'): return 'R5'
    if session.startswith('sixth'): return 'R6'
    return 'R7m' if session in ('seventh_13', 'seventh_14') else 'R7o'


def build():
    """Rows: x (104), hand, drum, session, group, g (peak g). Cached."""
    f = OUT/'dataset.joblib'
    if f.exists(): return joblib.load(f)
    import soft_hit_experiment as S
    C = S.C; rows = []
    for r in C['baseline']:
        rows.append(dict(x=np.asarray(r['x'], np.float32), hand=r['hand'], drum=r['drum'], session=r['session'], id=str(r['id'])))
    for code in ['03', '04', '05', '13', '14']:
        L = json.loads((LAB/f'{code}_完成標記_20261021.json').read_text(encoding='utf-8'))['labels']
        rr, _ = S.rows_from_labels(f'seventh_{code}', L)
        # rows_from_labels drops ids; re-derive id by matching order is unnecessary for training
        for r in rr: rows.append(dict(x=np.asarray(r['x'], np.float32), hand=r['hand'], drum=r['drum'], session=f'seventh_{code}', id=''))
    for r in C['auto']:
        if r['session'] in [f'seventh_{i:02d}' for i in range(6, 12)]:
            rows.append(dict(x=np.asarray(r['x'], np.float32), hand=r['hand'], drum=r['drum'], session=r['session'], id=str(r['id'])))
    for r in rows:
        r['group'] = group_of(r['session']); r['g'] = float(np.expm1(r['x'][17]))
    tests = {k: dict(truth=C['tests'][k]['truth'], items=[dict(hand=c['peak']['hand'], t=c['peak']['peak_ms'], g=float(c['peak']['peak_g']),
                     x=np.asarray(c['x'], np.float32)) for c in C['tests'][k]['items']]) for k in TESTS}
    D = dict(rows=rows, tests=tests); joblib.dump(D, f); return D


class MLP:
    def __init__(self, classes, hidden=(512, 256, 128), epochs=120, lr=2e-3, wd=1e-3, drop=.2, smooth=.05, seed=0, batch=256):
        self.classes_ = list(classes); self.h = hidden; self.ep = epochs; self.lr = lr; self.wd = wd; self.drop = drop
        self.smooth = smooth; self.seed = seed; self.batch = batch

    def fit(self, X, y, w):
        torch.manual_seed(self.seed); np.random.seed(self.seed)
        self.mu = X.mean(0); self.sd = X.std(0)+1e-6
        Xt = torch.tensor((X-self.mu)/self.sd, dtype=torch.float32, device=DEV)
        yi = torch.tensor([self.classes_.index(v) for v in y], device=DEV); wt = torch.tensor(w, dtype=torch.float32, device=DEV)
        layers, d = [], X.shape[1]
        for hd in self.h: layers += [torch.nn.Linear(d, hd), torch.nn.BatchNorm1d(hd), torch.nn.GELU(), torch.nn.Dropout(self.drop)]; d = hd
        layers += [torch.nn.Linear(d, len(self.classes_))]
        self.net = torch.nn.Sequential(*layers).to(DEV)
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=self.wd)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=self.lr, total_steps=self.ep*((len(Xt)+self.batch-1)//self.batch))
        for _ in range(self.ep):
            self.net.train(); perm = torch.randperm(len(Xt), device=DEV)
            for i in range(0, len(Xt), self.batch):
                b = perm[i:i+self.batch]
                loss = (torch.nn.functional.cross_entropy(self.net(Xt[b]), yi[b], reduction='none', label_smoothing=self.smooth)*wt[b]).mean()
                opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        return self

    def predict_proba(self, X):
        self.net.eval()
        with torch.no_grad():
            return torch.softmax(self.net(torch.tensor((X-self.mu)/self.sd, dtype=torch.float32, device=DEV)), 1).cpu().numpy()


def weights(rows, cfg):
    c = Counter(r['drum'] for r in rows); w = []
    gs = Counter(r['group'] for r in rows)
    for r in rows:
        v = 1/np.sqrt(c[r['drum']]) if cfg.get('class_bal', 'sqrt') == 'sqrt' else (1/c[r['drum']] if cfg.get('class_bal') == 'full' else 1.0)
        if cfg.get('group_bal'): v *= 1/np.sqrt(gs[r['group']])
        v *= cfg.get('group_w', {}).get(r['group'], 1.0)
        w.append(v)
    w = np.array(w, np.float32); return w/w.mean()


def fit_hand(rows, cfg, seed):
    X = np.stack([r['x'] for r in rows]); y = [r['drum'] for r in rows]; w = weights(rows, cfg); classes = sorted(set(y))
    models = []
    if cfg.get('xgb', True):
        p = dict(n_estimators=600, max_depth=6, learning_rate=.05, subsample=.8, colsample_bytree=.6, reg_lambda=2.)
        p.update(cfg.get('xgb_params', {}))
        m = xgb.XGBClassifier(**p, device='cuda', tree_method='hist', random_state=seed)
        m.fit(X, np.array([classes.index(v) for v in y]), sample_weight=w); models.append(m)
    if cfg.get('mlp', True):
        models.append(MLP(classes, seed=seed, **cfg.get('mlp_params', {})).fit(X, y, w))
    return classes, models


def train(rows, cfg, seeds=(0, 1)):
    rows = [r for r in rows if r['g'] >= cfg.get('min_g', 0)]
    return [{h: fit_hand([r for r in rows if r['hand'] == h], cfg, s) for h in 'LR'} for s in seeds]


def proba(committee, hand, X):
    ps = []
    for member in committee:
        classes, models = member[hand]
        p = np.mean([m.predict_proba(X) for m in models], 0)
        full = np.zeros((len(X), len(DRUMS)))
        for j, c in enumerate(classes): full[:, DRUMS.index(c)] = p[:, j]
        ps.append(full)
    return np.mean(ps, 0)


def predict_items(committee, items):
    out = []
    for h in 'LR':
        it = [c for c in items if c['hand'] == h]
        if not it: continue
        p = proba(committee, h, np.stack([c['x'] for c in it]))
        out += [dict(t=c['t'], g=c['g'], drum=DRUMS[int(q.argmax())], score=float(q.max())) for c, q in zip(it, p)]
    return out


def score_pretest(truth, preds, thr=0.0):
    keep = [p for p in preds if p['score'] >= thr]
    tt = np.array([g['t'] for g in truth]); lo, hi = tt.min()-200, tt.max()+200
    keep = [p for p in keep if lo <= p['t'] <= hi]
    if not keep: return dict(correct=0, total=len(truth), out=0)
    pt = np.array([p['t'] for p in keep]); dt = np.abs(tt[:, None]-pt[None, :])
    cost = np.hstack([np.where(dt <= 80, dt, 1e9), np.full((len(tt), len(tt)), 1e6)])
    r, c = linear_sum_assignment(cost); pairs = {i: j for i, j in zip(r, c) if j < len(pt) and cost[i, j] <= 80}
    ok = sum(1 for i, j in pairs.items() if keep[j]['drum'] == truth[i]['drum'])
    return dict(correct=ok, total=len(truth), out=len(keep), matched=len(pairs))


def exam(committee, D, thrs=(0.0, 0.6, 0.8)):
    res = {}
    for k, t in D['tests'].items():
        preds = predict_items(committee, t['items'])
        res[k] = {str(th): score_pretest(t['truth'], preds, th) for th in thrs}
    summ = {}
    for th in thrs:
        th = str(th); rates = [res[k][th]['correct']/res[k][th]['total'] for k in res]
        prec = sum(res[k][th]['correct'] for k in res)/max(sum(res[k][th]['out'] for k in res), 1)
        summ[th] = dict(mean_rate=round(float(np.mean(rates)), 4), min_rate=round(float(np.min(rates)), 4), precision=round(prec, 4))
    return res, summ


def group_cv(rows, cfg, seeds=(0,), thr=0.6):
    out = {}
    for g in ['R4', 'R5', 'R6', 'R7o', 'R7m']:
        tr = [r for r in rows if r['group'] != g]; te = [r for r in rows if r['group'] == g]
        com = train(tr, cfg, seeds); ok = n = ans = okc = 0
        for h in 'LR':
            t = [r for r in te if r['hand'] == h]
            if not t: continue
            p = proba(com, h, np.stack([r['x'] for r in t]))
            pred = [DRUMS[int(q.argmax())] for q in p]; sc = p.max(1)
            ok += sum(a == r['drum'] for a, r in zip(pred, t)); n += len(t)
            m = sc >= thr; ans += int(m.sum()); okc += sum(a == r['drum'] for a, r, k in zip(pred, t, m) if k)
        out[g] = dict(acc=round(ok/n, 4), n=n, precision_at_thr=round(okc/max(ans, 1), 4), answered=round(ans/n, 4))
    out['mean_acc'] = round(float(np.mean([out[g]['acc'] for g in ['R4', 'R5', 'R6', 'R7o', 'R7m']])), 4)
    return out
