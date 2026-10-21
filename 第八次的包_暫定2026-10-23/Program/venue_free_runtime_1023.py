"""10/23 runtime: hit gate + venue-free committee + causal per-drum offset correction.

Per candidate peak (detector at 1.5 g, released at peak + 80 ms):
  1. hit gate: a learned classifier decides whether the stick really hit a drum, from the peak's
     shape (detector features), the 104 venue-free features, the other hand's motion within +-40 ms
     and the time / strength relative to the previous candidate of the same hand. Non-hits are dropped
     and do not enter any history.
  2. drum: venue-free committee with per-drum offset correction from EARLIER accepted hits of the same
     hand in this take (score >= 0.7, after >= 20 such hits, two passes). No labels, no future.
  3. pose continuity: if this hit's score is < 0.9 and the same hand's previous accepted hit was within
     600 ms with a zero-relative yaw/pitch within 16 deg, it is the same drum as that previous hit.
"""
from scipy.spatial.transform import Rotation as _Rot
from collections import deque

import numpy as np

import venue_free_runtime_1020 as base

CONTRACT = 'hologrip_venue_free_committee_1023_v1'
INPUT_DIMENSIONS = base.INPUT_DIMENSIONS
POSE_DTS = base.POSE_DTS
ROT = slice(83, 104)
feature = base.feature
probabilities = base.probabilities


CONTINUITY = dict(max_ms=600.0, max_deg=16.0, below_score=0.9)


def _yaw_pitch(x):
    v = np.asarray(x[ROT]).reshape(7, 3)[4]*np.pi
    return _Rot.from_rotvec(v).as_euler('ZYX', degrees=True)[:2]


def attach_boosters(bundle):
    base.attach_boosters(bundle)
    gate = bundle['hit_gate']; bst = bundle['_xgb'].Booster(); bst.load_model(bytearray(gate['xgb_json'])); bst.set_param({'nthread': 1})
    bundle['_hit_gate'] = bst
    return bundle


def hit_probability(bundle, g, det, x, ctx):
    d = np.asarray(det, np.float32) if len(det) else np.zeros(18, np.float32)
    v = np.r_[g, d, x, ctx].astype(np.float32)[None, :]
    return float(bundle['_hit_gate'].predict(bundle['_xgb'].DMatrix(v))[0])


class _Stats:
    """Running per-drum sums of raw rotation blocks of confident hits (one per pass)."""
    def __init__(self):
        self.sum = np.zeros((7, 21)); self.n = np.zeros(7); self.total = 0

    def add(self, rot, p, conf):
        if p.max() >= conf:
            d = int(p.argmax()); self.sum[d] += rot; self.n[d] += 1; self.total += 1

    def offset(self, tm):
        m = self.n > 0
        return (self.n[m, None]*tm[m]-self.sum[m]).sum(0)/self.n[m].sum()


class StreamEngine:
    def __init__(self, bundle):
        self.bundle = bundle; ad = bundle['adaptation']
        self.alpha, self.conf, self.minh = ad['alpha'], ad['confidence'], ad['min_confident_hits']
        self.gate_threshold = bundle['hit_gate']['threshold']
        self.classes = [str(c) for c in bundle['classes']]
        self.tm = {h: np.array([bundle['drum_rotation_means'].get(f'{h}|{d}', np.zeros(21)) for d in self.classes]) for h in 'LR'}
        self.history = {h: deque(maxlen=1024) for h in 'LR'}
        self.hit = {h: base.world.base.detector.HitStream(h, bundle['detector_config']) for h in 'LR'}
        self.stats = {h: [_Stats(), _Stats()] for h in 'LR'}
        self.last_candidate = {h: None for h in 'LR'}
        self.hand_mode = {h: None for h in 'LR'}; self.zero_yaw = {h: None for h in 'LR'}; self.zero_pitch = {h: None for h in 'LR'}
        self.heading_distance = {h: None for h in 'LR'}; self.count = 0; self.adapted = {h: 0 for h in 'LR'}
        self.rejected = {h: 0 for h in 'LR'}; self.previous = {h: None for h in 'LR'}; self.continued = {h: 0 for h in 'LR'}

    def _classify(self, hand, x):
        st = self.stats[hand]; p0 = probabilities(self.bundle, x[None, :], hand)[0]; p1 = p0
        if st[0].total >= self.minh:
            x1 = x.copy(); x1[ROT] = x[ROT]+self.alpha*st[0].offset(self.tm[hand]); p1 = probabilities(self.bundle, x1[None, :], hand)[0]
        pf = p1; x2 = x
        if st[1].total >= self.minh:
            x2 = x.copy(); x2[ROT] = x[ROT]+self.alpha*st[1].offset(self.tm[hand]); pf = probabilities(self.bundle, x2[None, :], hand)[0]
            self.adapted[hand] += 1
        st[0].add(x[ROT], p0, self.conf); st[1].add(x[ROT], p1, self.conf)
        return pf, _yaw_pitch(x2)

    def _other_peak(self, hand, t):
        other = 'R' if hand == 'L' else 'L'
        vals = [s['mag'] for s in self.history[other] if abs(s['t']-t) <= 40]
        return max(vals) if vals else 1.0

    def push(self, hand, sample):
        self.zero_yaw[hand] = (sample['absolute_yaw']-sample['yaw']+180) % 360-180
        self.zero_pitch[hand] = sample['absolute_pitch']-sample['pitch']
        self.hand_mode[hand] = 'venue_free_adaptive'
        self.history[hand].append(sample)
        peaks = self.hit[hand].push(sample)
        if not peaks: return []
        rows = list(self.history[hand])
        prepared = base.world.base.legacy.prepare({hand: rows}); first = rows[0]['index']; output = []
        for peak in sorted(peaks, key=lambda p: p['peak_ms']):
            local = {**peak, 'peak_index': int(peak['peak_index'])-first, 'available_index': len(rows)-1}
            vector = feature(prepared, local, self.zero_yaw[hand], self.zero_pitch[hand])
            if vector is None: continue
            t, g = float(peak['peak_ms']), float(peak['peak_g']); pv = self.last_candidate[hand]
            ctx = np.array([self._other_peak(hand, t), np.log1p(min(t-pv[0], 3000)) if pv else np.log1p(3000), g/pv[1] if pv else 1.0], np.float32)
            self.last_candidate[hand] = (t, g)
            ph = hit_probability(self.bundle, g, peak.get('feature', []), vector, ctx)
            if ph < self.gate_threshold:
                self.rejected[hand] += 1; continue
            q, yp = self._classify(hand, vector); top = np.sort(q); label = int(q.argmax()); prev = self.previous[hand]
            if (prev is not None and q.max() < CONTINUITY['below_score'] and t-prev[0] <= CONTINUITY['max_ms']
                    and float(np.hypot(*(yp-prev[1]))) <= CONTINUITY['max_deg']):
                label = prev[2]; self.continued[hand] += 1
            self.previous[hand] = (t, yp, label)
            self.count += 1
            output.append(dict(**{k: v for k, v in peak.items() if k != 'feature'}, prediction_id=self.count,
                               direction_mode='venue_free_adaptive', zero_yaw_deg=self.zero_yaw[hand], heading_distance_deg=None,
                               drum=self.classes[label], model_score=float(q.max()), model_margin=float(top[-1]-top[-2]),
                               hit_probability=ph))
        return output
