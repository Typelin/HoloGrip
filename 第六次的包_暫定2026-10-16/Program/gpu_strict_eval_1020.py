"""GPU strict evaluation (user standard).

- Pretests NEVER in training (train = songs + four-drum + clear single-hand).
- Each hit judged alone from its own -240..+80 ms window plus the drumhead zero recorded
  before the take started. No other hits, no whole-recording statistics, no MIDI.
- No compass heading: trajectory features are relative to the hand's own heading; pose
  features are relative to the recorded zero (taken in the same room) -> venue-free.
Feature sets:
  N   heading-free trajectory only
  Z   N + pose relative to zero (Euler: yaw-zero (sin,cos), pitch-zero, roll) at 7 times
  A2  N + mount-invariant rotation relative to zero-anchor (training uses the SAME zero-based
      anchor as testing, so train/test conditions match)
Models: XGBoost (CUDA), PyTorch MLP (CUDA), and their probability average.
"""
from __future__ import annotations
import json, sys, time
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import torch
import xgboost as xgb
from scipy.spatial.transform import Rotation as R

PROG = Path(__file__).resolve().parent
sys.path.insert(0, str(PROG))
OUT = PROG.parent/'Data/鼓組對準模型_1020'
DEV = 'cuda'


def wrap(a): return (np.asarray(a)+180.0) % 360.0-180.0


def load():
    zfile = OUT/'zero_offsets.json'
    if zfile.exists():
        return json.loads(zfile.read_text(encoding='utf-8'))
    import pose_experiments_1020 as E
    z = E.zero_offsets()
    zfile.write_text(json.dumps(z), encoding='utf-8')
    return z


def pose_feats(P, z, kind):
    """P: (n,7,3) raw yaw/pitch/roll; z: (yaw0, pitch0, roll0)."""
    n = len(P)
    if kind == 'Z':
        y = np.radians(wrap(P[:, :, 0]-z[0])); p = (P[:, :, 1]-z[1])/90.0; r = wrap(P[:, :, 2])/180.0
        return np.hstack([np.sin(y), np.cos(y), p, r])
    if kind == 'A2':
        A = R.from_euler('ZYX', [[z[0], z[1], 0.0]], degrees=True)
        Hd = R.from_euler('Z', [-z[0]], degrees=True)
        Rt = R.from_euler('ZYX', P.reshape(-1, 3), degrees=True)
        return ((Hd*Rt*A.inv()*Hd.inv()).as_rotvec().reshape(n, -1))/np.pi
    return np.zeros((n, 0))


def rot_batch(P, Z):
    """P (n,7,3) raw poses, Z (n,3) per-sample anchor (yaw,pitch,roll) -> (n,21) mount-invariant rotation."""
    n, k = P.shape[0], P.shape[1]
    A = R.from_euler('ZYX', np.repeat(Z, k, axis=0), degrees=True)
    Hd = R.from_euler('Z', np.repeat(-Z[:, 0], k)[:, None], degrees=True)
    Rt = R.from_euler('ZYX', P.reshape(-1, 3), degrees=True)
    return ((Hd*Rt*A.inv()*Hd.inv()).as_rotvec().reshape(n, -1))/np.pi


def build_train(tr, z, aug_deg, copies, subset, seed=0):
    """A2 features; optional anchor jitter (simulates different drumhead-zero gestures)."""
    rng = np.random.default_rng(seed)
    out = {}
    for h in 'LR':
        E_ = [e for e in tr if e['hand'] == h]
        HF = np.array([e['hf'] for e in E_], dtype=np.float32); P = np.array([e['pose'] for e in E_])
        Z0 = np.array([z[e['session']][h] or (0, 0, 0) for e in E_], dtype=float); Z0[:, 2] = 0.0
        y = np.array([e['drum'] for e in E_])
        Xs, ys = [], []
        for c in range(copies+1):
            Z = Z0.copy()
            if c > 0:
                Z[:, 0] += rng.uniform(-aug_deg, aug_deg, len(Z)); Z[:, 1] += rng.uniform(-aug_deg/2, aug_deg/2, len(Z)); Z[:, 2] += rng.uniform(-aug_deg/2, aug_deg/2, len(Z))
            rb = rot_batch(P, Z)
            Xs.append(np.hstack([sub(HF, subset), rb]).astype(np.float32)); ys.append(y)
        out[h] = (np.vstack(Xs), np.concatenate(ys))
    return out


def sub(HF, subset):
    if subset == 'all':
        return HF
    # trajectory summary only: last 66 dims = 4 windows x 15 stats + scale + hand (+ integral snaps 21)
    return HF[:, -(21+2+60):]


class TorchMLP:
    def __init__(self, classes, hidden=(512, 256, 128), epochs=120, lr=2e-3, wd=1e-3, drop=0.2, seed=0):
        self.classes_ = np.array(classes); self.h = hidden; self.ep = epochs; self.lr = lr; self.wd = wd; self.drop = drop; self.seed = seed

    def fit(self, X, y, w):
        torch.manual_seed(self.seed)
        self.mu = X.mean(0); self.sd = X.std(0)+1e-6
        Xt = torch.tensor((X-self.mu)/self.sd, dtype=torch.float32, device=DEV)
        yi = torch.tensor([list(self.classes_).index(v) for v in y], device=DEV)
        wt = torch.tensor(w, dtype=torch.float32, device=DEV)
        layers, d = [], X.shape[1]
        for hdim in self.h:
            layers += [torch.nn.Linear(d, hdim), torch.nn.BatchNorm1d(hdim), torch.nn.GELU(), torch.nn.Dropout(self.drop)]; d = hdim
        layers += [torch.nn.Linear(d, len(self.classes_))]
        self.net = torch.nn.Sequential(*layers).to(DEV)
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=self.wd)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, self.ep)
        n = len(Xt)
        for _ in range(self.ep):
            self.net.train(); perm = torch.randperm(n, device=DEV)
            for i in range(0, n, 256):
                b = perm[i:i+256]
                loss = (torch.nn.functional.cross_entropy(self.net(Xt[b]), yi[b], reduction='none', label_smoothing=0.05)*wt[b]).mean()
                opt.zero_grad(); loss.backward(); opt.step()
            sched.step()
        return self

    def predict_proba(self, X):
        self.net.eval()
        with torch.no_grad():
            return torch.softmax(self.net(torch.tensor((X-self.mu)/self.sd, dtype=torch.float32, device=DEV)), 1).cpu().numpy()


def fit_models(X, y, w, kind, seed=0):
    classes = sorted(set(y))
    out = {}
    if kind in ('xgb', 'ens'):
        m = xgb.XGBClassifier(n_estimators=600, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.6,
                              reg_lambda=2.0, device='cuda', tree_method='hist', random_state=seed)
        m.fit(X, np.array([classes.index(v) for v in y]), sample_weight=w); out['xgb'] = m
    if kind in ('mlp', 'ens'):
        out['mlp'] = TorchMLP(classes, seed=seed).fit(X, y, w)
    return classes, out


def proba(models, X):
    ps = [m.predict_proba(X) for m in models.values()]
    return sum(ps)/len(ps)


def main2():
    t0 = time.perf_counter()
    C = joblib.load(OUT/'pose_cache.joblib'); z = load()
    import pose_experiments_1020 as E
    configs = [(a, cp, ss, k) for a in (0, 10, 20) for cp in ((0,) if a == 0 else (4,)) for ss in ('all', 'summary') for k in ('xgb', 'mlp', 'ens')]
    report = {}
    for aug, copies, subset, kind in configs:
        data = build_train(C['train'], z, aug, copies, subset)
        models = {}
        for h in 'LR':
            X, y = data[h]; c = Counter(y); w = np.array([1/np.sqrt(c[v]) for v in y], dtype=np.float32); w /= w.mean()
            models[h] = fit_models(X, y, w, kind)
        res = {}
        for k, t in E.TESTS.items():
            g = set(E.GHOST[k]['ghosts']); S = dict(t['session'], truth=[x for x in t['session']['truth'] if x['id'] not in g])
            tt = np.array([float(x['t']) for x in S['truth']]); lo, hi = tt.min()-200, tt.max()+200
            preds = []
            for h in 'LR':
                items = [c for c in C['peaks'][k] if c['peak']['hand'] == h and lo <= c['peak']['peak_ms'] <= hi]
                if not items: continue
                P = np.array([c['pose'] for c in items]); Z = np.array([list(z[k][h][:2])+[0.0]]*len(items), dtype=float)
                X = np.hstack([sub(np.array([c['hf'] for c in items], dtype=np.float32), subset), rot_batch(P, Z)]).astype(np.float32)
                classes, ms = models[h]
                d = np.array(classes)[proba(ms, X).argmax(1)]
                preds += [dict(**{q: v for q, v in c['peak'].items() if q != 'feature'}, drum=str(x), model_score=1.0) for c, x in zip(items, d)]
            preds.sort(key=lambda p: p['peak_ms'])
            sc = E.H.first.scoring.score(S, preds)[0]
            res[k] = round(sc['correct']/sc['total'], 3)
        key = f'A2 aug{aug} {subset} {kind}'
        report[key] = res
        print(key, res, 'min', min(res.values()), round(time.perf_counter()-t0), 's', flush=True)
    (OUT/'gpu_strict_results_2.json').write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding='utf-8')


def main3():
    """Seed/augmentation committees on A2+summary (strict standard)."""
    t0 = time.perf_counter()
    C = joblib.load(OUT/'pose_cache.joblib'); z = load()
    import pose_experiments_1020 as E
    members = {}
    for aug in (0, 10, 15):
        for seed in (0, 1, 2):
            data = build_train(C['train'], z, aug, 0 if aug == 0 else 4, 'summary', seed=seed)
            ms = {}
            for h in 'LR':
                X, y = data[h]; c = Counter(y); w = np.array([1/np.sqrt(c[v]) for v in y], dtype=np.float32); w /= w.mean()
                ms[h] = fit_models(X, y, w, 'ens', seed=seed)
            members[(aug, seed)] = ms
        print('trained aug', aug, round(time.perf_counter()-t0), 's', flush=True)
    def evaluate(keys):
        res = {}
        for k, t in E.TESTS.items():
            g = set(E.GHOST[k]['ghosts']); S = dict(t['session'], truth=[x for x in t['session']['truth'] if x['id'] not in g])
            tt = np.array([float(x['t']) for x in S['truth']]); lo, hi = tt.min()-200, tt.max()+200
            preds = []
            for h in 'LR':
                items = [c for c in C['peaks'][k] if c['peak']['hand'] == h and lo <= c['peak']['peak_ms'] <= hi]
                if not items: continue
                P = np.array([c['pose'] for c in items]); Z = np.array([list(z[k][h][:2])+[0.0]]*len(items), dtype=float)
                X = np.hstack([sub(np.array([c['hf'] for c in items], dtype=np.float32), 'summary'), rot_batch(P, Z)]).astype(np.float32)
                classes = members[keys[0]][h][0]
                pr = sum(proba(members[kk][h][1], X) for kk in keys)/len(keys)
                d = np.array(classes)[pr.argmax(1)]
                preds += [dict(**{q: v for q, v in c['peak'].items() if q != 'feature'}, drum=str(x), model_score=1.0) for c, x in zip(items, d)]
            preds.sort(key=lambda p: p['peak_ms'])
            sc = E.H.first.scoring.score(S, preds)[0]
            res[k] = round(sc['correct']/sc['total'], 3)
        return res
    report = {}
    combos = {'aug0 x3seeds': [(0, s) for s in range(3)], 'aug10 x3seeds': [(10, s) for s in range(3)], 'aug15 x3seeds': [(15, s) for s in range(3)],
              'aug0+10 x3': [(a, s) for a in (0, 10) for s in range(3)], 'aug10+15 x3': [(a, s) for a in (10, 15) for s in range(3)],
              'aug0+10+15 x3': [(a, s) for a in (0, 10, 15) for s in range(3)]}
    for name, keys in combos.items():
        r = evaluate(keys); report[name] = r
        print(name, r, 'min', min(r.values()), flush=True)
    (OUT/'gpu_strict_results_3.json').write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding='utf-8')
    joblib.dump(dict(members=members, note='A2+summary committees; pretests never trained'), OUT/'gpu_strict_committee.joblib', compress=3)


def main():
    t0 = time.perf_counter()
    C = joblib.load(OUT/'pose_cache.joblib')
    z = load()
    tests_meta = joblib.load(OUT.parent/'..'/'..'/'第六次的包_暫定2026-10-16'/'Data'/'鼓組對準模型_1020'/'pose_cache.joblib') if False else None
    import pose_experiments_1020 as E              # scoring, ghost list, test sessions
    print('loaded', round(time.perf_counter()-t0), 's', flush=True)
    tr = C['train']
    feats_sets = sys.argv[1].split(',') if len(sys.argv) > 1 else ['N', 'Z', 'A2']
    kinds = sys.argv[2].split(',') if len(sys.argv) > 2 else ['xgb', 'mlp', 'ens']
    report = {}
    for fs in feats_sets:
        for h in 'LR':
            pass
        Xtr = {h: [] for h in 'LR'}; ytr = {h: [] for h in 'LR'}
        for e in tr:
            pf = pose_feats(e['pose'][None], z[e['session']][e['hand']] or (0, 0, 0), fs)[0]
            Xtr[e['hand']].append(np.r_[e['hf'], pf]); ytr[e['hand']].append(e['drum'])
        for kind in kinds:
            models = {}
            for h in 'LR':
                X = np.array(Xtr[h], dtype=np.float32); y = np.array(ytr[h])
                c = Counter(y); w = np.array([1/np.sqrt(c[v]) for v in y], dtype=np.float32); w /= w.mean()
                models[h] = fit_models(X, y, w, kind)
            res = {}
            for k, t in E.TESTS.items():
                g = set(E.GHOST[k]['ghosts']); S = dict(t['session'], truth=[x for x in t['session']['truth'] if x['id'] not in g])
                tt = np.array([float(x['t']) for x in S['truth']]); lo, hi = tt.min()-200, tt.max()+200
                preds = []
                for h in 'LR':
                    items = [c for c in C['peaks'][k] if c['peak']['hand'] == h and lo <= c['peak']['peak_ms'] <= hi]
                    if not items: continue
                    P = np.array([c['pose'] for c in items])
                    X = np.hstack([np.array([c['hf'] for c in items]), pose_feats(P, z[k][h] or (0, 0, 0), fs)]).astype(np.float32)
                    classes, ms = models[h]
                    d = np.array(classes)[proba(ms, X).argmax(1)]
                    preds += [dict(**{q: v for q, v in c['peak'].items() if q != 'feature'}, drum=str(x), model_score=1.0) for c, x in zip(items, d)]
                preds.sort(key=lambda p: p['peak_ms'])
                sc = E.H.first.scoring.score(S, preds)[0]
                res[k] = round(sc['correct']/sc['total'], 3)
            report[f'{fs}|{kind}'] = res
            print(fs, kind, res, round(time.perf_counter()-t0), 's', flush=True)
    (OUT/'gpu_strict_results.json').write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding='utf-8')


if __name__ == '__main__':
    {'grid2': main2, 'grid3': main3}.get(sys.argv[1] if len(sys.argv) > 1 else '', main)()
