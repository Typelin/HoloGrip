"""Fast pretest experiments on cached committees (cache_models_1027.py).

Field-equivalent online decoding per hand (causal: every correction uses earlier hits only):
  committee  p = (1-w_rc) * raw-trained + w_rc * re-centred
  anchor     automatic snare anchor (first 5 hits: snare in top-k before correction, poses within deg)
  layout     'shift'  per-drum offset correction (current field method, 2 passes)
             'affine' shift + one scale factor of the rotation block around the drum means
                      (fits squeezed / stretched kits); needs >= 2 drums seen
  continuity same as field, parameters configurable
Scores: per day total and every hand x drum cell (pretests only; never trained).
"""
from __future__ import annotations
import json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import joblib
from scipy.spatial.transform import Rotation as _R

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lodo5 as L5
K = L5.K
SNARE = K.DRUMS.index('小鼓'); ROT = slice(83, 104)
DAYS = ['R4', 'R5', 'R6', 'R7', 'R8']
CACHE = HERE/'out/cache'
DEFAULT = dict(w_rc=0.5, anchor_top=2, anchor_deg=8.0, layout='shift', conf=0.7, minh=20, cont_deg=16.0, cont_score=0.9, cont_top=2,
               scale_min=0.6, scale_max=1.4, min_drums=2, pos_w=0.0, pos_sigma=10.0, pos_in_stats=False, cont_margin=1.0, pos_all=False, anchor_any=False, anchor_minp=0.5, pos_sigma_pitch=None, layout_mix=False, mix_temp=1.0, mix_conf=0.8, mix_min=0.0,
               auto_zero=False, az_step=5.0, az_range=90.0, az_sigma=40.0, az_min=5, az_window=0, az_score='logmax', az_freeze=0, az_mode='conf', yf_conf=0.7, yf_min=5, yf_round=0.0, yf_thr=0.0, yf_agree=0.0, yf_band=15.0, az_consistent=False, yf_drums=1, az_verify=False, az_margin=0.0, az_persist=0, yf_max=999.0)


def pos_prior(Y, tmY, sigma, sigma_p=None):
    """Gaussian likelihood of the corrected hit pose (yaw/pitch at the hit) around each drum's training mean."""
    d = Y[:, None, :]-tmY[None, :, :]; d[..., 0] = (d[..., 0]+180) % 360-180  # yaw is circular
    sp = sigma if sigma_p is None else sigma_p
    d2 = (d[..., 0]/sigma)**2+(d[..., 1]/sp)**2
    return np.exp(-d2/2)


def combine(p, Y, tmY, o):
    if o['pos_w'] <= 0: return p
    q = p*pos_prior(Y, tmY, o['pos_sigma'], o['pos_sigma_pitch'])**o['pos_w']+1e-12
    return q/q.sum(1, keepdims=True)


def load(tag):
    M = joblib.load(CACHE/f'{tag}.joblib'); M['_tag'] = tag; return M


_LAY = {}


def mix_prior(M, h, p, Y, o):
    """Causal mixture of per-day layout templates: each template is weighted by how well the earlier
    confident hits of this take (their pose and decided drum) fit it."""
    lay = _LAY.setdefault(M['_tag'], joblib.load(CACHE/f"layouts_{M['_tag']}.joblib"))
    days = list(lay); muY = np.stack([L5.LF.yp(np.hstack([np.zeros((7, 83), np.float32), lay[D][h]['mean'].astype(np.float32)])) for D in days])
    sy = o['pos_sigma']; sp = o['pos_sigma_pitch'] or sy
    def lik(y):   # (layouts, drums)
        d = y[None, None, :]-muY; d[..., 0] = (d[..., 0]+180) % 360-180
        return np.exp(-((d[..., 0]/sy)**2+(d[..., 1]/sp)**2)/2)
    ll = np.zeros(len(days)); out = p.copy()
    for i in range(len(p)):
        w = np.exp((ll-ll.max())/o['mix_temp']); w /= w.sum(); Lk = lik(Y[i])
        q = p[i]*((w[:, None]*Lk).sum(0))**o['pos_w']+1e-12; q /= q.sum(); out[i] = q
        if q.max() >= o['mix_conf']: ll += np.log(Lk[:, int(q.argmax())]+1e-9)
    return out


def rotate_zero(X, delta):
    """Features as if the drumhead zero yaw had been delta degrees off: R_rel -> Rz(-delta) * R_rel."""
    if not delta: return X
    X = X.copy(); r = _R.from_rotvec(X[:, ROT].reshape(-1, 3).astype(np.float64)*np.pi)
    X[:, ROT] = ((_R.from_euler('Z', -delta, degrees=True)*r).as_rotvec()/np.pi).reshape(len(X), 21).astype(np.float32)
    return X


def proba(M, h, X, w):
    out = 0
    if w < 1: out = out+(1-w)*K.proba(M['com_raw'], h, X)
    if w > 0: out = out+w*K.proba(M['com_rc'], h, X)
    return out


def correction(sums, cnt, tm, o):
    """Returns f(x_rot) mapping observed rotation blocks into the training frame, from earlier hits."""
    m = cnt > 0
    obs = sums[m]/cnt[m, None]; tgt = tm[m]; w = cnt[m]
    t_shift = (w[:, None]*(tgt-obs)).sum(0)/w.sum()
    if o['layout'] == 'shift' or m.sum() < o['min_drums']:
        return lambda r: r+t_shift
    mo = (w[:, None]*obs).sum(0)/w.sum(); mt = (w[:, None]*tgt).sum(0)/w.sum()
    a, b = obs-mo, tgt-mt
    den = (w*(a*a).sum(1)).sum()
    s = float((w*(a*b).sum(1)).sum()/den) if den > 1e-9 else 1.0
    s = min(max(s, o['scale_min']), o['scale_max'])
    return lambda r: mt+s*(r-mo)


def stats(X, p, anchor, o):
    """Causal per-hit correction functions (one per hit) from earlier hits."""
    n = len(X); pred = p.argmax(1); ok = p.max(1) >= o['conf']
    sums = np.zeros((7, 21)); cnt = np.zeros(7); na = 0; fs = [None]*n
    return pred, ok, sums, cnt, na, fs


def auto_zero(M, h, X, o):
    """Causal zero-direction estimate per hit: the candidate rotation c that makes the hits so far
    (this one included) most confidently classified, with a Gaussian preference for small c."""
    n = len(X); cand = np.arange(-o['az_range'], o['az_range']+1e-6, o['az_step'])
    S = np.zeros((len(cand), n))
    for k, c in enumerate(cand):
        p = proba(M, h, rotate_zero(X, c), o['w_rc'])
        S[k] = np.log(p.max(1)+1e-9) if o['az_score'] == 'logmax' else p.max(1)
    prior = -0.5*(cand/o['az_sigma'])**2
    cs = np.cumsum(S, 1); best = np.zeros(n)
    for i in range(n):
        if i+1 < o['az_min']: continue
        if o['az_freeze'] and i >= o['az_freeze']: best[i] = best[i-1]; continue
        tot = cs[:, i]-(cs[:, i-o['az_window']] if o['az_window'] and i >= o['az_window'] else 0)
        best[i] = cand[int(np.argmax(tot+prior))]
    Xc = X.copy()
    for c in np.unique(best):
        idx = best == c; Xc[idx] = rotate_zero(X[idx], c)
    return Xc, best


_YF = {}


def yawfree_zero(M, h, X, o):
    """Causal zero-direction estimate from the heading-free classifier: for hits it is confident about,
    (training yaw of that drum) - (hit yaw) estimates the zero error; circular median of hits so far."""
    import yawfree_1027 as YF
    clf = _YF.setdefault(M['_tag'], joblib.load(CACHE/f"yawfree_{M['_tag']}.joblib"))[h]
    pf = clf.predict_proba(YF.yawfree(X)); n = len(X)
    tmY = L5.LF.yp(np.hstack([np.zeros((7, 83), np.float32), M['tm'][h].astype(np.float32)]))
    Y = L5.LF.yp(X); d = (tmY[pf.argmax(1), 0]-Y[:, 0]+180) % 360-180; ok = pf.max(1) >= o['yf_conf']
    est = np.zeros(n); vals = []; vdr = []
    for i in range(n):
        if ok[i]: vals.append(d[i]); vdr.append(int(pf[i].argmax()))
        if len(vals) >= o['yf_min'] and o['yf_agree'] > 0:
            v = np.array(vals); grid = np.arange(-180, 180, 5.0)
            cnt = np.array([(np.abs((v-g+180) % 360-180) <= o['yf_band']).sum() for g in grid]); k = int(cnt.argmax())
            inb = np.abs((v-grid[k]+180) % 360-180) <= o['yf_band']
            if cnt[k] >= o['yf_agree']*len(v) and len(set(np.array(vdr)[inb])) >= o['yf_drums']:
                sel = v[inb]; m = float(np.median(sel))
                m = round(m/o['yf_round'])*o['yf_round'] if o['yf_round'] else m
                est[i] = m if o['yf_thr'] <= abs(m) <= o['yf_max'] else 0.0
            else: est[i] = est[i-1] if i else 0.0
        elif len(vals) >= o['yf_min']:
            a = np.radians(vals); m = np.degrees(np.arctan2(np.median(np.sin(a)), np.median(np.cos(a))))
            m = round(m/o['yf_round'])*o['yf_round'] if o['yf_round'] else m
            est[i] = m if abs(m) >= o['yf_thr'] else 0.0
    Xc = X.copy()
    for i in range(n):
        if est[i]: Xc[i:i+1] = rotate_zero(X[i:i+1], -est[i])
    return Xc, est


def run_hand(M, h, X, t, o):
    if o['auto_zero'] and o['az_consistent']:
        # every hit is decided with the whole history rotated by the estimate current at that hit
        _, est = yawfree_zero(M, h, X, o); lab = np.zeros(len(X), int); sc = np.zeros(len(X)); anc = False
        if o['az_persist']:
            # use an estimate only after it has stayed the same for az_persist consecutive hits
            raw = est.copy(); run = 0; cur = 0.0
            for i in range(len(raw)):
                run = run+1 if i and raw[i] == raw[i-1] else 1
                if run >= o['az_persist']: cur = raw[i]
                est[i] = cur
        if o['az_verify']:
            # keep a correction only if the full committee is more confident over the hits so far than without it
            base = np.cumsum(np.log(proba(M, h, X, o['w_rc']).max(1)+1e-9)); cache = {}
            for i, e in enumerate(est):
                if not e: continue
                if e not in cache: cache[e] = np.cumsum(np.log(proba(M, h, rotate_zero(X, -e), o['w_rc']).max(1)+1e-9))
                if (cache[e][i]-base[i])/(i+1) <= o['az_margin']: est[i] = 0.0
        for e in np.unique(est):
            l2, a2 = run_hand(M, h, rotate_zero(X, -e), t, dict(o, auto_zero=False)); idx = est == e
            lab[idx] = l2[idx]; sc[idx] = globals()['LAST_SCORE'][idx]; anc = anc or a2
        globals()['LAST_SCORE'] = sc
        return lab, anc
    w = o['w_rc']; tm = M['tm'][h]; n = len(X)
    if o['auto_zero']: X, _ = (yawfree_zero if o['az_mode'] == 'yawfree' else auto_zero)(M, h, X, o)
    tmY = L5.LF.yp(np.hstack([np.zeros((7, 83), np.float32), tm.astype(np.float32)]))
    p0 = proba(M, h, X, w)
    # automatic anchor
    anchor = set(); adrum = SNARE
    if n >= 5:
        Y = L5.LF.yp(X[:5]); med = np.median(Y, 0); tight = np.all(np.hypot(*(Y-med).T) <= o['anchor_deg'])
        if tight and all(SNARE in np.argsort(-p0[i])[:o['anchor_top']] for i in range(5)): anchor = set(range(5))
        elif tight and o['anchor_any']:
            top = p0[:5].argmax(1)
            if np.all(top == top[0]) and p0[:5, top[0]].mean() >= o['anchor_minp']: anchor = set(range(5)); adrum = int(top[0])
    p = p0
    for _ in range(2):
        pred = p.argmax(1); ok = p.max(1) >= o['conf']
        sums = np.zeros((7, 21)); cnt = np.zeros(7); na = 0; Xc = X.copy(); valid = np.zeros(n, bool)
        for i in range(n):
            if cnt.sum() >= o['minh'] or na >= 5:
                f = correction(sums, cnt, tm, o); Xc[i, ROT] = f(X[i, ROT]); valid[i] = True
            if i in anchor: sums[adrum] += X[i, ROT]; cnt[adrum] += 1; na += 1
            elif ok[i]: sums[pred[i]] += X[i, ROT]; cnt[pred[i]] += 1
        pc = proba(M, h, Xc, w); p = np.where(valid[:, None], pc, p)
        if o['pos_in_stats']: p = np.where(valid[:, None], combine(p, L5.LF.yp(Xc), tmY, o), p)
    Y = L5.LF.yp(Xc)
    if o['layout_mix']: p = mix_prior(M, h, p, Y, o)
    elif not o['pos_in_stats']: p = combine(p, Y, tmY, o) if o['pos_all'] else np.where(valid[:, None], combine(p, Y, tmY, o), p)
    lab = p.argmax(1).copy(); sc = p.max(1); srt = np.sort(p, 1); margin = srt[:, -1]-srt[:, -2]
    for i in range(1, n):
        if (sc[i] < o['cont_score'] and margin[i] < o['cont_margin'] and t[i]-t[i-1] <= 600 and np.hypot(*(Y[i]-Y[i-1])) <= o['cont_deg']
                and lab[i-1] in np.argsort(-p[i])[:o['cont_top']]):
            lab[i] = lab[i-1]
    globals()['LAST_SCORE'] = p[np.arange(n), lab]
    return lab, bool(anchor)


def evaluate(tag, opts=None, days=None):
    o = dict(DEFAULT, **(opts or {})); M = load(tag); res = {}
    for nm, d in M['items'].items():
        D = L5.day_of(nm)
        if days and D not in days: continue
        P = []
        for h in 'LR':
            lab, _ = run_hand(M, h, rotate_zero(d[h]['X'], o.get('zero_err', 0)), d[h]['t'], o)
            P += [dict(t=float(ti), peak_ms=float(ti), hand=h, drum=K.DRUMS[int(l)]) for ti, l in zip(d[h]['t'], lab)]
        truth = M['truth'][nm]; pairs = L5.match(truth, P)
        r = res.setdefault(D, dict(ok=0, n=0, cells=defaultdict(lambda: [0, 0])))
        for i, g in enumerate(truth):
            j = pairs.get(i); good = j is not None and (g['hand'] is None or P[j]['hand'] == g['hand']) and P[j]['drum'] == g['drum']
            r['ok'] += good; r['n'] += 1
            if g['hand']: c = r['cells'][f"{g['hand']}|{g['drum']}"]; c[0] += good; c[1] += 1
    return res


def summary(res):
    ok = sum(v['ok'] for v in res.values()); n = sum(v['n'] for v in res.values())
    below = {D: {k: f'{a}/{b}' for k, (a, b) in sorted(v['cells'].items()) if a/b < .85} for D, v in res.items()}
    worst = min(v['ok']/v['n'] for v in res.values())
    return dict(total=round(ok/n, 4), worst_day=round(worst, 4), days={D: round(v['ok']/v['n'], 4) for D, v in sorted(res.items())},
                n_below=sum(len(b) for b in below.values()), below=below)


if __name__ == '__main__':
    opts = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
    print(json.dumps(summary(evaluate(sys.argv[1], opts)), ensure_ascii=False))
