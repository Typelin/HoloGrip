"""Fast experiments on pose_cache: session normalisation x feature set x model.

Offsets per (recording, hand) for yaw / pitch / roll:
  oracle   DIAGNOSTIC ONLY: circular/median pose of labeled snare hits (answers)
  hist2d   label-free: 2-D (yaw, pitch) presence histogram of detected hits aligned to the
           training kit map; roll offset by 1-D alignment the same way
Pretests are never in training or in the kit map. Scoring: full Raw peaks inside the
MIDI span, ghost-cleaned answers, pretest scorer from the field pipeline.
"""
from __future__ import annotations
import json
import sys
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from scipy.ndimage import gaussian_filter
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

PROG = Path(__file__).resolve().parent
sys.path.insert(0, str(PROG))
import heading_free_model_1019 as H

OUT = PROG.parent/'Data/鼓組對準模型_1020'
C = joblib.load(OUT/'pose_cache.joblib')
SESS = joblib.load(H.first.OLD_CACHE)
BASE = joblib.load(H.first.OUT/'特徵快取.joblib')
PRED_LOG = {}
GHOST = json.loads((H.OUT/'幽靈鼓名單_規則固定.json').read_text(encoding='utf-8'))
TRAIN_NAMES = [s['name'] for s in SESS]
TESTS = {'pretest:'+t['session']['name']: t for t in BASE['tests']}


def wrap(a): return (np.asarray(a)+180.0) % 360.0-180.0


def circ_mean(deg):
    r = np.radians(deg); return float(np.degrees(np.arctan2(np.sin(r).mean(), np.cos(r).mean())))


# ---------------- offsets ----------------
def oracle_offsets():
    off = {}
    tr = [e for e in C['train']]
    for name in TRAIN_NAMES:
        off[name] = {}
        for h in 'LR':
            P = np.array([e['pose'][4] for e in tr if e['session'] == name and e['hand'] == h and e['drum'] == '小鼓'])
            off[name][h] = None if len(P) < 3 else (circ_mean(P[:, 0]), float(np.median(P[:, 1])), float(np.median(P[:, 2])))
    # recordings without own snare for a hand borrow the nearest same-day recording (same gloves,
    # raw yaw/pitch/roll frame is unaffected by re-zeroing): sixth day order S02,S04,S05,S06,S07,S08
    order = ['sixth_S02', 'sixth_S04', 'sixth_S05', 'sixth_S06', 'sixth_S07', 'sixth_S08']
    for k in order:
        for h in 'LR':
            if off[k][h] is None:
                i = order.index(k)
                donors = sorted((abs(order.index(d)-i), d) for d in order if d != k and off[d][h] is not None)
                if donors: off[k][h] = off[donors[0][1]][h]
    # one day-level anchor for the sixth day: in-song snare of S02 and S08 (the single-hand takes only have
    # slow opening taps, whose wrist posture differs by ~18 deg pitch from playing posture)
    for h in 'LR':
        songs = [off[k][h] for k in ('sixth_S02', 'sixth_S08') if off[k][h] is not None]
        day = (circ_mean([v[0] for v in songs]), float(np.mean([v[1] for v in songs])), float(np.mean([v[2] for v in songs])))
        for k in order: off[k][h] = day
    for k, t in TESTS.items():
        off[k] = {}
        sn = [float(g['t']) for g in t['session']['truth'] if g['drum'] == '小鼓']
        for h in 'LR':
            P = [c['pose'][4] for c in C['peaks'][k] if c['peak']['hand'] == h and any(abs(c['peak']['peak_ms']-x) <= 60 for x in sn)]
            P = np.array(P)
            off[k][h] = None if len(P) < 3 else (circ_mean(P[:, 0]), float(np.median(P[:, 1])), float(np.median(P[:, 2])))
    return off


def hist2(yaw, pitch):
    h = np.zeros((180, 90))
    for a, b in zip(yaw, pitch):
        h[int(np.floor((a % 360)/2)) % 180, int(np.clip((b+90)//2, 0, 89))] += 1
    h = np.sqrt(h)                                   # presence-weighted, less sensitive to drum frequency
    h = gaussian_filter(h, (2.0, 1.5), mode=('wrap', 'nearest'))
    return h/max(h.sum(), 1e-9)


def align2(h, ref, pitch_max=10, yaw_max=None):
    best = (-1, 0, 0)
    F = np.fft.fft(h, axis=0); R = np.fft.fft(ref, axis=0)
    for dp in range(-pitch_max, pitch_max+1):
        r = np.roll(ref, dp, axis=1)
        if dp > 0: r[:, :dp] = 0
        if dp < 0: r[:, dp:] = 0
        c = np.real(np.fft.ifft(F*np.conj(np.fft.fft(r, axis=0)), axis=0)).sum(axis=1)
        if yaw_max is not None:
            sh = (np.arange(180)*2+180) % 360-180
            c = np.where(np.abs(sh) <= yaw_max, c, -np.inf)
        k = int(np.argmax(c))
        if c[k] > best[0]: best = (c[k], k*2, dp*2)
    return best[1], best[2]                       # yaw shift deg, pitch shift deg


def hist_offsets(ref_name='fifth_song03'):
    poses = {k: {h: np.array([c['pose'][4] for c in v if c['peak']['hand'] == h]) for h in 'LR'} for k, v in C['peaks'].items()}
    hs = {k: {h: (hist2(p[h][:, 0], p[h][:, 1]) if len(p[h]) > 20 else None) for h in 'LR'} for k, p in poses.items()}
    ref = {h: hs[ref_name][h].copy() for h in 'LR'}
    for _ in range(3):
        sh = {k: {h: align2(hs[k][h], ref[h]) for h in 'LR' if hs[k][h] is not None} for k in TRAIN_NAMES}
        for h in 'LR':
            acc = np.zeros_like(ref[h])
            for k in TRAIN_NAMES:
                if h in sh[k]:
                    dy, dp = sh[k][h]
                    acc += np.roll(np.roll(hs[k][h], -dy//2, axis=0), -dp//2, axis=1)
            ref[h] = acc/acc.sum()
    off = {}
    rref = {h: np.median(np.concatenate([poses[k][h][:, 2] for k in TRAIN_NAMES if len(poses[k][h]) > 20])) for h in 'LR'}
    for k in poses:
        off[k] = {}
        for h in 'LR':
            if hs[k][h] is None: off[k][h] = None; continue
            dy, dp = align2(hs[k][h], ref[h])
            off[k][h] = (float(dy), float(dp), float(np.median(poses[k][h][:, 2])-rref[h]))
    return off


def zero_offsets():
    """Label-free, venue-free: the recorded drumhead zero (raw minus calibrated yaw/pitch).
    Roll has no recorded zero; roll offset 0."""
    off = {}
    def z(hands, raw):
        H.world.enrich(hands, raw); prep = H.world.base.legacy.prepare(hands)
        out = {}
        for h in 'LR':
            rows = prep[h]['rows']
            if not rows: out[h] = None; continue
            r = rows[len(rows)//2]
            out[h] = (float(wrap(r['absolute_yaw']-r['yaw'])), float(r['absolute_pitch']-r['pitch']), 0.0)
        return out
    for s in SESS: off[s['name']] = z(s['hands'], s['raw'])
    for k, t in TESTS.items(): off[k] = z(t['hands'], t['session']['raw'])
    return off


def zero_hist_offsets(ref_name='fifth_song03', yaw_max=None, pitch_max=None):
    import os
    yaw_max = yaw_max if yaw_max is not None else float(os.environ.get('HG_YAWMAX', 40))
    pitch_max = pitch_max if pitch_max is not None else int(os.environ.get('HG_PITCHMAX', 10))
    """Label-free: recorded drumhead zero as coarse offset, then presence-histogram alignment of
    zero-relative hit poses to the training kit map within +/-yaw_max deg and +/-pitch_max deg."""
    z = zero_offsets()
    poses = {}
    for k, v in C['peaks'].items():
        poses[k] = {}
        for h in 'LR':
            P = np.array([c['pose'][4] for c in v if c['peak']['hand'] == h])
            if len(P) and z[k][h] is not None:
                P = P.copy(); P[:, 0] = wrap(P[:, 0]-z[k][h][0]); P[:, 1] = P[:, 1]-z[k][h][1]
            poses[k][h] = P
    hs = {k: {h: (hist2(p[h][:, 0], p[h][:, 1]) if len(p[h]) > 20 else None) for h in 'LR'} for k, p in poses.items()}
    ref = {h: hs[ref_name][h].copy() for h in 'LR'}
    for _ in range(3):
        sh = {k: {h: align2(hs[k][h], ref[h], pitch_max, yaw_max) for h in 'LR' if hs[k][h] is not None} for k in TRAIN_NAMES}
        for h in 'LR':
            acc = np.zeros_like(ref[h])
            for k in TRAIN_NAMES:
                if h in sh[k]:
                    dy, dp = sh[k][h]
                    acc += np.roll(np.roll(hs[k][h], -int(dy)//2, axis=0), -int(dp)//2, axis=1)
            ref[h] = acc/acc.sum()
    global KITREF
    KITREF = dict(ref=ref, yaw_max=yaw_max, pitch_max=pitch_max)
    off = {}
    for k in poses:
        off[k] = {}
        for h in 'LR':
            if hs[k][h] is None or z[k][h] is None: off[k][h] = z[k][h]; continue
            dy, dp = align2(hs[k][h], ref[h], pitch_max, yaw_max)
            dy = float(wrap(dy))
            off[k][h] = (float(wrap(z[k][h][0]+dy)), z[k][h][1]+dp, 0.0)
    return off


def lprior_offsets(yaw_max=None):
    """Label-free. Left hand: zero + histogram refinement (stable). Right hand: prior yaw =
    left offset + median (right-left) offset difference measured on TRAINING recordings with
    the same label-free method, then histogram refinement within +/-yaw_max around that prior.
    Pretests never contribute to the constant."""
    import os
    yaw_max = yaw_max if yaw_max is not None else float(os.environ.get('HG_RMAX', 30))
    base = zero_hist_offsets()
    diffs = [float(wrap(base[k]['R'][0]-base[k]['L'][0])) for k in TRAIN_NAMES if base[k]['L'] and base[k]['R']]
    d0 = circ_mean(diffs)
    z = zero_offsets()
    # right-hand kit map in the zero-hist frame from training recordings
    poses = {k: np.array([c['pose'][4] for c in C['peaks'][k] if c['peak']['hand'] == 'R']) for k in C['peaks']}
    def rel(k, off):
        P = poses[k].copy(); P[:, 0] = wrap(P[:, 0]-off[0]); P[:, 1] = P[:, 1]-off[1]; return P
    ref = None
    for k in TRAIN_NAMES:
        if base[k]['R'] is None or len(poses[k]) < 20: continue
        h = hist2(*rel(k, base[k]['R'])[:, :2].T); ref = h if ref is None else ref+h
    ref /= ref.sum()
    out = {}
    for k in C['peaks']:
        out[k] = dict(base[k])
        if base[k]['L'] is None or len(poses[k]) < 20 or z[k]['R'] is None: continue
        prior = (float(wrap(base[k]['L'][0]+d0)), z[k]['R'][1], 0.0)
        dy, dp = align2(hist2(*rel(k, prior)[:, :2].T), ref, 6, yaw_max)
        out[k]['R'] = (float(wrap(prior[0]+wrap(dy))), prior[1]+dp, 0.0)
    return out


# ---------------- features / model ----------------
from scipy.spatial.transform import Rotation as _R


def rot_kit(pose, o):
    """Mount-invariant: rotation from the anchor (snare) pose to each pose, in the anchor heading frame."""
    A = _R.from_euler('ZYX', [o[0], o[1], o[2]], degrees=True)
    Hd = _R.from_euler('Z', -o[0], degrees=True)
    Rt = _R.from_euler('ZYX', pose[:, [0, 1, 2]], degrees=True)
    rel = Hd*Rt*A.inv()*Hd.inv()
    return rel.as_rotvec().ravel()/np.pi


def feat(hf, pose, hand, o, fset):
    if o is None: o = (0.0, 0.0, 0.0)
    if fset in ('rot', 'rotonly'):
        kit = np.r_[rot_kit(pose, o), float(hand == 'R')]
        return kit if fset == 'rotonly' else np.r_[hf, kit]
    y = np.radians(wrap(pose[:, 0]-o[0])); p = (pose[:, 1]-o[1])/90.0; r = wrap(pose[:, 2]-o[2])/180.0
    kit = np.r_[np.sin(y), np.cos(y), p, r, float(hand == 'R')]
    return kit if fset == 'pose' else np.r_[hf, kit]


class Avg:
    """Average class probabilities of several fitted classifiers (same classes)."""
    def __init__(self, models, weights=None):
        self.models = models; self.w = weights or [1.0]*len(models)
        self.classes_ = models[0].classes_
        assert all(list(m.classes_) == list(self.classes_) for m in models)
    def predict_proba(self, X):
        return sum(w*m.predict_proba(X) for w, m in zip(self.w, self.models))/sum(self.w)
    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(1)]


def fit(X, y, hands, kind):
    if kind.startswith('ens'):
        parts = {'ens': ['gbdt', 'et'], 'ens3': ['gbdt', 'et', 'gbdt_reg']}[kind]
        fitted = [fit(X, y, hands, k) for k in parts]
        return {h: Avg([f[h] for f in fitted]) for h in 'LR'}
    cell = Counter(zip(hands, y)); w = np.array([1/np.sqrt(cell[(h, z)]) for h, z in zip(hands, y)]); w /= w.mean()
    models = {}
    for h in 'LR':
        m = hands == h
        if kind == 'gbdt_reg':
            mdl = HistGradientBoostingClassifier(max_iter=500, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=40, l2_regularization=3.0, random_state=0)
            mdl.fit(X[m], y[m], sample_weight=w[m])
        elif kind == 'et':
            from sklearn.ensemble import ExtraTreesClassifier
            mdl = ExtraTreesClassifier(600, min_samples_leaf=3, max_features='sqrt', random_state=int(__import__('os').environ.get('HG_SEED', 0)), n_jobs=4)
            mdl.fit(X[m], y[m], sample_weight=w[m])
        elif kind == 'gbdt':
            mdl = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, max_leaf_nodes=31, l2_regularization=1.0, random_state=int(__import__('os').environ.get('HG_SEED', 0)))
            mdl.fit(X[m], y[m], sample_weight=w[m])
        else:
            mdl = Pipeline([('s', StandardScaler()), ('m', MLPClassifier((256, 128), alpha=.3, random_state=20261019, max_iter=450, n_iter_no_change=25, tol=1e-4, batch_size=128))])
            mdl.fit(X[m], y[m], m__sample_weight=w[m])
        models[h] = mdl
    return models


def predict(models, items, offsets_k, fset):
    out = []
    for c in items:
        h = c['peak']['hand']
        x = feat(c['hf'], c['pose'], h, offsets_k[h], fset)
        out.append((c, x))
    preds = []
    for h in 'LR':
        sel = [(c, x) for c, x in out if c['peak']['hand'] == h]
        if not sel: continue
        P = models[h].predict(np.array([x for _, x in sel]))
        preds += [dict(**{k: v for k, v in c['peak'].items() if k != 'feature'}, drum=str(d), model_score=1.0) for (c, _), d in zip(sel, P)]
    return sorted(preds, key=lambda p: p['peak_ms'])


def evaluate(models, offsets, fset):
    res = {}
    for k, t in TESTS.items():
        g = set(GHOST[k]['ghosts']); S = dict(t['session'], truth=[x for x in t['session']['truth'] if x['id'] not in g])
        tt = np.array([float(x['t']) for x in S['truth']])
        items = [c for c in C['peaks'][k] if tt.min()-200 <= c['peak']['peak_ms'] <= tt.max()+200]
        sc = H.first.scoring.score(S, predict(models, items, offsets[k], fset))[0]
        res[k] = round(sc['correct']/sc['total'], 3)
    for s in SESS:
        g = set(GHOST[s['name']]['ghosts'])
        b, _, _ = H.first.training.assess(dict(s, truth=[x for x in s['truth'] if x['id'] not in g]), predict(models, C['peaks'][s['name']], offsets[s['name']], fset))
        res[s['name']] = round(b['rate'], 3)
    return res


def pretest_train_items(k, offsets):
    """Labeled training items from a pretest (used only when that pretest is NOT the test)."""
    t = TESTS[k]; g = set(GHOST[k]['ghosts'])
    items = []
    used = set()
    for a in t['session']['truth']:
        if a['id'] in g: continue
        cands = [c for c in C['peaks'][k] if abs(c['peak']['peak_ms']-float(a['t'])) <= 60 and id(c) not in used
                 and (('hand' not in a) or c['peak']['hand'] == a['hand'])]
        if not cands: continue
        c = min(cands, key=lambda c: abs(c['peak']['peak_ms']-float(a['t']))); used.add(id(c))
        items.append(dict(session=k, hand=c['peak']['hand'], drum=a['drum'], hf=c['hf'], pose=c['pose']))
    return items


def run_lopo(offsets, fset, kind, pre_weight=None):
    import os
    pre_weight = int(os.environ.get('HG_PREW', 3)) if pre_weight is None else pre_weight
    """Leave-one-pretest-out: train = songs + the other two pretests; test = held pretest only."""
    res = {}
    for held in TESTS:
        tr = list(C['train'])
        extra = [e for k in TESTS if k != held for e in pretest_train_items(k, offsets)]
        tr += extra*pre_weight
        X = np.array([feat(e['hf'], e['pose'], e['hand'], offsets[e['session']][e['hand']], fset) for e in tr])
        y = np.array([e['drum'] for e in tr]); hands = np.array([e['hand'] for e in tr])
        with threadpool_limits(limits=1):
            models = fit(X, y, hands, kind)
        t = TESTS[held]; g = set(GHOST[held]['ghosts']); S = dict(t['session'], truth=[x for x in t['session']['truth'] if x['id'] not in g])
        tt = np.array([float(x['t']) for x in S['truth']])
        items = [c for c in C['peaks'][held] if tt.min()-200 <= c['peak']['peak_ms'] <= tt.max()+200]
        sc = H.first.scoring.score(S, predict(models, items, offsets[held], fset))[0]
        res[held] = round(sc['correct']/sc['total'], 3)
    return res


def run_lopo_ml(fset, kind, pre_weight=None):
    """Training recordings (incl. the two non-held pretests) aligned with THEIR labels (legit:
    training data). Held pretest: label-free maximum-confidence search of yaw/pitch offset per
    hand around the label-free zero+hist estimate (no answers of the held pretest are read)."""
    import os
    pre_weight = int(os.environ.get('HG_PREW', 3)) if pre_weight is None else pre_weight
    span = float(os.environ.get('HG_SPAN', 30)); step = float(os.environ.get('HG_STEP', 2.5))
    orc = oracle_offsets(); zh = zero_hist_offsets()
    # frame conversion zero+hist -> label-aligned, measured on TRAINING recordings only
    conv = {}
    for h in 'LR':
        dy = [float(wrap(orc[k][h][0]-zh[k][h][0])) for k in TRAIN_NAMES if orc[k][h] and zh[k][h]]
        dp = [orc[k][h][1]-zh[k][h][1] for k in TRAIN_NAMES if orc[k][h] and zh[k][h]]
        dr = [orc[k][h][2]-zh[k][h][2] for k in TRAIN_NAMES if orc[k][h] and zh[k][h]]
        conv[h] = (circ_mean(dy), float(np.median(dp)), float(np.median(dr)))
    start = {k: {h: (None if zh[k][h] is None else (float(wrap(zh[k][h][0]+conv[h][0])), zh[k][h][1]+conv[h][1], zh[k][h][2]+conv[h][2])) for h in 'LR'} for k in zh}
    if os.environ.get('HG_LROLL', '1') == '1':
        # left hand roll anchor = median roll of the recording's left hits + training-measured constant
        med = {k: float(np.median([c['pose'][4][2] for c in C['peaks'][k] if c['peak']['hand'] == 'L'])) for k in C['peaks'] if any(c['peak']['hand'] == 'L' for c in C['peaks'][k])}
        dl = float(np.median([orc[k]['L'][2]-med[k] for k in TRAIN_NAMES if k in med and orc[k]['L'] and k not in ('sixth_S04', 'sixth_S05', 'sixth_S06', 'sixth_S07')]))
        for k in start:
            if start[k]['L'] is not None and k in med:
                start[k]['L'] = (start[k]['L'][0], start[k]['L'][1], med[k]+dl)
    res, chosen = {}, {}
    for held in TESTS:
        offs = {k: v for k, v in orc.items()}
        tr = list(C['train'])
        extra = [e for k in TESTS if k != held for e in pretest_train_items(k, offs)]
        tr += extra*pre_weight
        X = np.array([feat(e['hf'], e['pose'], e['hand'], offs[e['session']][e['hand']], fset) for e in tr])
        y = np.array([e['drum'] for e in tr]); hands = np.array([e['hand'] for e in tr])
        with threadpool_limits(limits=1):
            models = fit(X, y, hands, kind)
        t = TESTS[held]; g = set(GHOST[held]['ghosts']); S = dict(t['session'], truth=[x for x in t['session']['truth'] if x['id'] not in g])
        tt = np.array([float(x['t']) for x in S['truth']])
        items = [c for c in C['peaks'][held] if tt.min()-200 <= c['peak']['peak_ms'] <= tt.max()+200]
        test_off = {}
        for h in 'LR':
            hi = [c for c in items if c['peak']['hand'] == h]
            if not hi: test_off[h] = start[held][h]; continue
            s0 = start[held][h] or (0.0, 0.0, 0.0)
            best = (-1e9, s0)
            rspan = float(os.environ.get('HG_RSPAN', 0))
            for dy in np.arange(-span, span+1e-9, step):
                for dp in np.arange(-10, 10.1, 2.5):
                    for dr in (np.arange(-rspan, rspan+1e-9, 5.0) if rspan else [0.0]):
                        o = (float(wrap(s0[0]+dy)), s0[1]+dp, s0[2]+dr)
                        P = models[h].predict_proba(np.array([feat(c['hf'], c['pose'], h, o, fset) for c in hi]))
                        sc = float(np.mean(np.log(P.max(axis=1)+1e-9)))
                        if sc > best[0]: best = (sc, o)
            test_off[h] = best[1]
        offs[held] = test_off
        chosen[held] = {h: (None if orc[held][h] is None or test_off[h] is None else round(float(wrap(test_off[h][0]-orc[held][h][0])), 1)) for h in 'LR'}
        preds = predict(models, items, offs[held], fset)
        PRED_LOG[held] = [(round(p['peak_ms'], 3), p['hand'], p['drum']) for p in preds]
        sc = H.first.scoring.score(S, preds)[0]
        res[held] = round(sc['correct']/sc['total'], 3)
    return res, chosen


def causal_offset(h, hits, z, conv, model, fset, rec_every=10, min_hits=None):
    import os
    min_hits = int(os.environ.get('HG_MINHITS', 20)) if min_hits is None else min_hits
    """Offset for hand h using ONLY the given (already played) hits of this recording."""
    s0 = (float(wrap(z[0]+conv[0])), z[1]+conv[1], z[2]+conv[2])
    if len(hits) < min_hits:
        return s0
    P = np.array([c['pose'][4] for c in hits]).copy()
    P[:, 0] = wrap(P[:, 0]-z[0]); P[:, 1] = P[:, 1]-z[1]
    dy, dp = align2(hist2(P[:, 0], P[:, 1]), KITREF['ref'][h], KITREF['pitch_max'], KITREF['yaw_max'])
    zh = (float(wrap(z[0]+wrap(dy))), z[1]+dp, 0.0)
    s0 = (float(wrap(zh[0]+conv[0])), zh[1]+conv[1], zh[2]+conv[2])
    best = (-1e9, s0)
    for dyy in np.arange(-30, 30.1, 2.5):
        for dpp in np.arange(-10, 10.1, 2.5):
            o = (float(wrap(s0[0]+dyy)), s0[1]+dpp, s0[2])
            Pp = model.predict_proba(np.array([feat(c['hf'], c['pose'], h, o, fset) for c in hits]))
            sc = float(np.mean(np.log(Pp.max(axis=1)+1e-9)))
            if sc > best[0]: best = (sc, o)
    return best[1]


def run_lopo_causal(fset, kind, pre_weight=None, rec_every=10):
    """Real-time simulation: each hit is classified with an alignment estimated only from hits
    that had already happened (all detected hits of the recording since recording start)."""
    import os
    pre_weight = int(os.environ.get('HG_PREW', 6)) if pre_weight is None else pre_weight
    orc = oracle_offsets(); zh = zero_hist_offsets(); z = zero_offsets()
    conv = {}
    for h in 'LR':
        pairs = [(orc[k][h], zh[k][h]) for k in TRAIN_NAMES if orc[k][h] and zh[k][h]]
        conv[h] = (circ_mean([float(wrap(a[0]-b[0])) for a, b in pairs]), float(np.median([a[1]-b[1] for a, b in pairs])), float(np.median([a[2]-b[2] for a, b in pairs])))
    res, curve = {}, {}
    for held in TESTS:
        tr = list(C['train']) + [e for k in TESTS if k != held for e in pretest_train_items(k, orc)]*pre_weight
        X = np.array([feat(e['hf'], e['pose'], e['hand'], orc[e['session']][e['hand']], fset) for e in tr])
        y = np.array([e['drum'] for e in tr]); hands = np.array([e['hand'] for e in tr])
        with threadpool_limits(limits=1):
            models = fit(X, y, hands, kind)
        t = TESTS[held]; g = set(GHOST[held]['ghosts']); S = dict(t['session'], truth=[x for x in t['session']['truth'] if x['id'] not in g])
        tt = np.array([float(x['t']) for x in S['truth']]); lo, hi = tt.min()-200, tt.max()+200
        allhits = sorted(C['peaks'][held], key=lambda c: c['peak']['peak_ms'])
        preds = []
        cache = {'L': (-1, None), 'R': (-1, None)}
        for i, c in enumerate(allhits):
            h = c['peak']['hand']
            past = [q for q in allhits[:i+1] if q['peak']['hand'] == h]       # up to and including this hit
            n = len(past)
            if cache[h][1] is None or (n-cache[h][0]) >= rec_every:
                cache[h] = (n, causal_offset(h, past, z[held][h], conv[h], models[h], fset))
            if lo <= c['peak']['peak_ms'] <= hi:
                x = feat(c['hf'], c['pose'], h, cache[h][1], fset)
                d = models[h].predict(np.array([x]))[0]
                preds.append(dict(**{k: v for k, v in c['peak'].items() if k != 'feature'}, drum=str(d), model_score=1.0))
        sc, rows = H.first.scoring.score(S, preds)[:2]
        res[held] = round(sc['correct']/sc['total'], 3)
        # accuracy by answer order (first 20 / rest)
        order = sorted(rows, key=lambda r: float(r['t']))
        curve[held] = dict(first20=round(np.mean([r['status'] == 'correct' for r in order[:20]]), 3),
                           rest=round(np.mean([r['status'] == 'correct' for r in order[20:]]), 3),
                           hits_before_midi_span=sum(1 for c in allhits if c['peak']['peak_ms'] < lo))
    return res, curve


def run(offsets, fset, kind):
    tr = C['train']
    X = np.array([feat(e['hf'], e['pose'], e['hand'], offsets[e['session']][e['hand']], fset) for e in tr])
    y = np.array([e['drum'] for e in tr]); hands = np.array([e['hand'] for e in tr])
    with threadpool_limits(limits=1):
        models = fit(X, y, hands, kind)
    return evaluate(models, offsets, fset)


if __name__ == '__main__':
    mode, fset, kind = sys.argv[1], sys.argv[2], sys.argv[3]
    offsets = {} if mode in ('mlsearch', 'causal') else {'oracle': oracle_offsets, 'zero': zero_offsets, 'zerohist': zero_hist_offsets, 'lprior': lprior_offsets}.get(mode, hist_offsets)()
    import os
    if mode == 'causal':
        r, cv = run_lopo_causal(fset, kind)
        print('CAUSAL', fset, kind, 'prew', os.environ.get('HG_PREW', 6), 'minhits', os.environ.get('HG_MINHITS', 20), r, cv, flush=True)
        (OUT/f'causal_{fset}_{kind}_w{os.environ.get("HG_PREW", 6)}_m{os.environ.get("HG_MINHITS", 20)}.json').write_text(json.dumps(dict(results=r, curve=cv), ensure_ascii=False, indent=1), encoding='utf-8')
        sys.exit(0)
    if mode == 'mlsearch':
        r, ch = run_lopo_ml(fset, kind)
        print('LOPO-ML', 'seed', os.environ.get('HG_SEED', 0), fset, kind, os.environ.get('HG_PREW', 3), 'span', os.environ.get('HG_SPAN', 30), 'rspan', os.environ.get('HG_RSPAN', 0), r, 'yaw error vs label alignment', ch, flush=True)
        (OUT/f'lopo_ml_{fset}_{kind}_w{os.environ.get("HG_PREW", 3)}.json').write_text(json.dumps(dict(results=r, yaw_error=ch), ensure_ascii=False, indent=1), encoding='utf-8')
        sys.exit(0)
    if len(sys.argv) > 4 and sys.argv[4] == 'lopo':
        r = run_lopo(offsets, fset, kind)
        print('LOPO', mode, fset, kind, os.environ.get('HG_YAWMAX', 40), os.environ.get('HG_PITCHMAX', 10), os.environ.get('HG_PREW', 3), r, flush=True)
        import os
        tag = f"y{os.environ.get('HG_YAWMAX', 40)}_p{os.environ.get('HG_PITCHMAX', 10)}_w{os.environ.get('HG_PREW', 3)}"
        (OUT/f'lopo_{mode}_{fset}_{kind}_{tag}.json').write_text(json.dumps(r, ensure_ascii=False, indent=1), encoding='utf-8')
        sys.exit(0)
    r = run(offsets, fset, kind)
    print(mode, fset, kind, {k: v for k, v in r.items() if k.startswith('pretest')}, '| min train', min(v for k, v in r.items() if not k.startswith('pretest')), flush=True)
    (OUT/f'exp_{mode}_{fset}_{kind}.json').write_text(json.dumps(dict(offsets={k: v for k, v in offsets.items()}, results=r), ensure_ascii=False, indent=1, default=str), encoding='utf-8')
