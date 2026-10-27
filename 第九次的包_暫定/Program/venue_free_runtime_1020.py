"""Venue-free per-hit drum classifier (2026-10-20 strict committee).

Each hit is judged alone, as soon as the external detector releases it (peak + 80 ms):
  * trajectory summary in the hand's own heading frame (constant compass shifts cancel);
  * hand rotation relative to the drumhead zero recorded before this take
    (Hd * R(t) * A^-1 * Hd^-1, independent of room direction and glove mounting);
  * six-member committee (zero jitter 10/15 deg x 3 seeds), each = XGBoost + MLP per hand.
No other hits, no whole-recording statistics, no MIDI, no compass heading as a feature.
Pretests (round 4/5/6) were never in training.
"""
from bisect import bisect_left, bisect_right
from collections import deque

import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from scipy.special import erf

import absolute_world_motion_runtime_1019 as world

CONTRACT = 'hologrip_venue_free_committee_1020_v1'
POSE_DTS = (-120, -80, -40, -20, 0, 20, 40)
SUMMARY_DIMENSIONS = 83
POSE_DIMENSIONS = 21
INPUT_DIMENSIONS = SUMMARY_DIMENSIONS+POSE_DIMENSIONS


def trajectory_summary(prepared, peak):
    """Last 83 dims of heading_free_model_1019.heading_free: integral snaps (last 17 of 21) + scale, hand, 4 windows x 16 stats."""
    rows = prepared[peak['hand']]['rows']; times = prepared[peak['hand']]['times']
    available = min(int(peak['available_index']), len(rows)-1)
    idx = peak.get('peak_index')
    if idx is None or not 0 <= int(idx) <= available or abs(rows[int(idx)]['t']-peak['peak_ms']) > .01:
        delta = np.abs(times[:available+1]-peak['peak_ms']); ties = np.flatnonzero(delta == delta.min())
        idx = max(ties, key=lambda i: rows[int(i)]['mag'])
    idx = int(idx); before = 240
    sc = rows[idx]['sensor_t']
    lo = bisect_left(rows, sc-before-30, 0, available+1, key=lambda r: r['sensor_t'])
    hi = bisect_right(rows, sc+80, 0, available+1, key=lambda r: r['sensor_t'])
    sel = rows[lo:hi]
    if len(sel) < 6: return None
    clock = np.array([r['sensor_t']-sc for r in sel]); _, u = np.unique(clock, return_index=True)
    sel = [sel[int(i)] for i in u]; clock = clock[u]
    if len(clock) < 6 or np.any(np.diff(clock) <= 0): return None
    grid = np.arange(-before, 81., 10.); clipped = np.clip(grid, clock[0], clock[-1])
    acc = np.array([[r['ax'], r['ay'], r['az']] for r in sel])
    ang = np.array([[r['absolute_yaw'], r['absolute_pitch'], r['roll']] for r in sel])
    body = np.column_stack([np.interp(grid, clock, acc[:, j]) for j in range(3)])
    rot = Slerp(clock, Rotation.from_euler('ZYX', ang, degrees=True))(clipped)
    c = int(np.argmin(np.abs(grid)))
    yaw_c = rot[c].as_euler('ZYX', degrees=True)[0]
    H = Rotation.from_euler('Z', -yaw_c, degrees=True)          # hand's own heading at the hit
    lin = H.apply(rot.apply(body)-np.array([0., 0., 1.]))
    av = H.apply(np.vstack([np.zeros(3), (rot[1:]*rot[:-1].inv()).as_rotvec()/.01]))
    mag = np.linalg.norm(body, axis=1); scale = max(1., float(np.linalg.norm(acc, axis=1).max()))
    snap = np.array([int(np.argmin(abs(grid-t))) for t in [-160, -120, -80, -40, 0, 40, 80]])
    integ = np.cumsum(lin, axis=0)*.01/scale
    scal = [np.log1p(scale), float(peak['hand'] == 'R')]
    for f0, f1 in [(-before, -120), (-120, -40), (-40, 40), (40, 80)]:
        m = (grid >= f0) & (grid <= f1); a, w, g = lin[m], av[m], mag[m]
        scal.extend(np.r_[a.mean(0)/scale, a.std(0)/scale, w.mean(0)/50., w.std(0)/50., np.percentile(g, [10, 50, 90])/scale, np.abs(g-1).sum()/scale])
    v = np.r_[integ[snap].ravel(), scal][-SUMMARY_DIMENSIONS:]
    return v if np.isfinite(v).all() else None


def pose_path(prepared, hand, t, available):
    rows = prepared[hand]['rows']; times = prepared[hand]['times'][:available+1]
    out = []
    for dt in POSE_DTS:
        r = rows[int(np.argmin(np.abs(times-(t+dt))))]
        out.append([r['absolute_yaw'], r['absolute_pitch'], r['roll']])
    return np.array(out, dtype=np.float64)


def zero_relative_rotation(pose, zero_yaw, zero_pitch):
    """(7,3) raw yaw/pitch/roll -> (21,) rotation relative to the drumhead zero, heading removed."""
    A = Rotation.from_euler('ZYX', [zero_yaw, zero_pitch, 0.0], degrees=True)
    Hd = Rotation.from_euler('Z', -zero_yaw, degrees=True)
    Rt = Rotation.from_euler('ZYX', pose, degrees=True)
    return ((Hd*Rt*A.inv()*Hd.inv()).as_rotvec().ravel())/np.pi


def feature(prepared, peak, zero_yaw, zero_pitch):
    summary = trajectory_summary(prepared, peak)
    if summary is None: return None
    available = min(int(peak['available_index']), len(prepared[peak['hand']]['rows'])-1)
    pose = pose_path(prepared, peak['hand'], peak['peak_ms'], available)
    v = np.r_[summary, zero_relative_rotation(pose, zero_yaw, zero_pitch)].astype(np.float32)
    return v if np.isfinite(v).all() else None


def mlp_proba(p, X):
    h = (X-p['mu'])/p['sd']
    for L in p['layers']:
        h = h@L['W']+L['b']
        if L['act']:
            h = (h-L['bn_mean'])/np.sqrt(L['bn_var']+L['bn_eps'])*L['bn_w']+L['bn_b']
            h = 0.5*h*(1.0+erf(h/np.sqrt(2.0)))
    h = h-h.max(1, keepdims=True); e = np.exp(h)
    return e/e.sum(1, keepdims=True)


def attach_boosters(bundle):
    import xgboost as xgb
    boosters = {}
    for key, member in bundle['portable'].items():
        for h in 'LR':
            bst = xgb.Booster(); bst.load_model(bytearray(member[h]['xgb_json'])); bst.set_param({'nthread': 1})
            boosters[key, h] = bst
    bundle['_boosters'] = boosters
    bundle['_xgb'] = xgb
    return bundle


def probabilities(bundle, X, hand):
    """Committee average of (XGBoost + MLP)/2 over all members; columns follow bundle['classes']."""
    X = np.asarray(X, dtype=np.float32); out = 0
    for key, member in bundle['portable'].items():
        px = bundle['_boosters'][key, hand].inplace_predict(X)
        out = out+(px+mlp_proba(member[hand]['mlp'], X))/2
    return out/len(bundle['portable'])


class StreamEngine:
    """Per-hand history + external hit detector; one prediction per released hit."""
    def __init__(self, bundle):
        self.bundle = bundle
        self.history = {h: deque(maxlen=1024) for h in 'LR'}
        self.hit = {h: world.base.detector.HitStream(h, bundle['detector_config']) for h in 'LR'}
        self.hand_mode = {h: None for h in 'LR'}
        self.zero_yaw = {h: None for h in 'LR'}
        self.zero_pitch = {h: None for h in 'LR'}
        self.heading_distance = {h: None for h in 'LR'}
        self.count = 0

    def push(self, hand, sample):
        # Drumhead zero in effect at this sample: raw minus calibrated angle. Constant for a
        # live take (zero is set before recording); older Raw files fill 0 until calibration.
        self.zero_yaw[hand] = (sample['absolute_yaw']-sample['yaw']+180) % 360-180
        self.zero_pitch[hand] = sample['absolute_pitch']-sample['pitch']
        self.hand_mode[hand] = 'venue_free'
        self.history[hand].append(sample)
        peaks = self.hit[hand].push(sample)
        if not peaks: return []
        rows = list(self.history[hand])
        prepared = world.base.legacy.prepare({hand: rows}); first = rows[0]['index']; output = []
        classes = [str(c) for c in self.bundle['classes']]
        for peak in peaks:
            local = {**peak, 'peak_index': int(peak['peak_index'])-first, 'available_index': len(rows)-1}
            vector = feature(prepared, local, self.zero_yaw[hand], self.zero_pitch[hand])
            if vector is None: continue
            q = probabilities(self.bundle, vector[None, :], hand)[0]
            self.count += 1; top = np.sort(q)
            output.append(dict(**{k: v for k, v in peak.items() if k != 'feature'}, prediction_id=self.count,
                               direction_mode='venue_free', zero_yaw_deg=self.zero_yaw[hand], heading_distance_deg=None,
                               drum=classes[int(q.argmax())], model_score=float(q.max()), model_margin=float(top[-1]-top[-2])))
        return output
